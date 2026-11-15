"""CLEAR360 v2: photos -> clear 3D scene.
SfM (pycolmap) -> Depth Anything V2 (flip-averaged, pinned to SfM points)
-> multi-view consistency filter -> TSDF fusion -> clean mesh.
pip install pycolmap open3d transformers torch opencv-python pillow   (Python 3.11/3.12)
python scripts/reconstruct_scene.py --photos my_photos --out data/scene_001
"""
import argparse
from pathlib import Path
import cv2, numpy as np, torch, pycolmap, open3d as o3d
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def prep(src, dst, max_side):
    dst.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        s = max_side / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        sharp = cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        items.append((p.stem, im, sharp))
    if not items:
        raise SystemExit("No photos found (use jpg/png; convert HEIC first)")
    med = np.median([i[2] for i in items])
    kept = 0
    for name, im, sharp in items:
        if sharp < 0.4 * med:
            print("skip blurry:", name)
            continue
        cv2.imwrite(str(dst / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
    print(f"{kept}/{len(items)} photos kept")
    return kept


def extract(db, img_dir, single):
    attempts = []
    try:
        o = pycolmap.FeatureExtractionOptions()
        o.sift.max_num_features = 16384
        attempts.append({"extraction_options": o})
    except Exception:
        pass
    attempts.append({})
    if single:
        attempts = [{**a, "camera_mode": pycolmap.CameraMode.SINGLE} for a in attempts] + attempts
    for kw in attempts:
        try:
            if db.exists():
                db.unlink()
            pycolmap.extract_features(str(db), str(img_dir), **kw)
            return
        except Exception as e:
            print("retrying feature extraction:", type(e).__name__)
    raise SystemExit("Feature extraction failed")


def run_sfm(img_dir, work, single):
    db = work / "db.db"
    extract(db, img_dir, single)
    pycolmap.match_exhaustive(str(db))
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(str(db), str(img_dir), str(sparse))
    if not maps:
        raise SystemExit("SfM failed: photos don't overlap. Retake with ~70% overlap.")
    best = max(maps.values(), key=lambda m: m.num_reg_images())
    n_in, n_reg = len(list(img_dir.glob("*.jpg"))), best.num_reg_images()
    print(f"registered {n_reg}/{n_in} photos | {len(best.points3D)} 3D points")
    try:
        print(f"reprojection error {best.compute_mean_reprojection_error():.2f}px | "
              f"track length {best.compute_mean_track_length():.1f}")
    except Exception:
        pass
    if n_reg < 0.7 * n_in:
        print("WARNING: many photos failed to register -> more overlap / more texture needed")
    bdir = sparse / "best"
    bdir.mkdir(exist_ok=True)
    best.write(str(bdir))
    und = work / "undist"
    pycolmap.undistort_images(str(und), str(bdir), str(img_dir))
    sp = und / "sparse"
    if not (sp / "cameras.bin").exists():
        sp = sp / "0"
    return pycolmap.Reconstruction(str(sp)), und / "images"


def pose(im):
    c = im.cam_from_world
    c = c() if callable(c) else c
    M = np.eye(4)
    M[:3] = c.matrix()
    return M  # world -> camera


@torch.no_grad()
def predict_disp(rgb, proc, net, dev, res):
    outs = []
    for img in (rgb, rgb[:, ::-1]):  # original + mirrored, then average
        inp = proc(images=Image.fromarray(np.ascontiguousarray(img)), return_tensors="pt",
                   size={"height": res, "width": res}).to(dev)
        p = net(**inp).predicted_depth[None]
        p = torch.nn.functional.interpolate(p, size=rgb.shape[:2], mode="bicubic",
                                            align_corners=False)[0, 0].cpu().numpy()
        outs.append(p)
    return 0.5 * (outs[0] + outs[1][:, ::-1])


def align(disp, uv, z):
    """Pin relative depth to real SfM points: global fit, then local correction."""
    h, w = disp.shape
    u = np.clip(uv[:, 0].astype(int), 0, w - 1)
    v = np.clip(uv[:, 1].astype(int), 0, h - 1)
    d, inv = disp[v, u], 1.0 / z
    keep = np.ones(len(d), bool)
    a = b = 0.0
    for _ in range(4):
        A = np.stack([d[keep], np.ones(keep.sum())], 1)
        a, b = np.linalg.lstsq(A, inv[keep], rcond=None)[0]
        r = a * d + b - inv
        new = np.abs(r) < 3 * np.median(np.abs(r[keep])) + 1e-9
        if new.sum() < 10:
            break
        keep = new
    R = np.zeros((h, w), np.float32)
    W = np.zeros((h, w), np.float32)
    R[v[keep], u[keep]] = (inv - (a * d + b))[keep]
    W[v[keep], u[keep]] = 1
    s = 0.05 * max(h, w)
    num, den = cv2.GaussianBlur(R, (0, 0), s), cv2.GaussianBlur(W, (0, 0), s)
    corr = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0)
    return a * disp + b + corr


def consistency(i, depths, views, nbrs, tol, need):
    """Keep a pixel only if neighbouring photos agree on its 3D position."""
    _, M, K, w, h, _, _ = views[i]
    d = depths[i]
    ys, xs = np.nonzero(d > 0)
    z = d[ys, xs].astype(np.float64)
    P = np.stack([(xs + .5 - K[0, 2]) * z / K[0, 0], (ys + .5 - K[1, 2]) * z / K[1, 1], z])
    Pw = M[:3, :3].T @ (P - M[:3, 3:4])
    ok = np.zeros(len(z), np.int32)
    for j in nbrs[i]:
        _, Mj, Kj, wj, hj, _, _ = views[j]
        Q = Mj[:3, :3] @ Pw + Mj[:3, 3:4]
        zq = np.maximum(Q[2], 1e-6)
        ui = np.floor(Kj[0, 0] * Q[0] / zq + Kj[0, 2]).astype(np.int64)
        vi = np.floor(Kj[1, 1] * Q[1] / zq + Kj[1, 2]).astype(np.int64)
        inside = (Q[2] > 1e-6) & (ui >= 0) & (ui < wj) & (vi >= 0) & (vi < hj)
        dj = np.zeros_like(zq)
        dj[inside] = depths[j][vi[inside], ui[inside]]
        ok += inside & (dj > 0) & (np.abs(dj - zq) < tol * zq)
    mask = np.zeros(d.shape, bool)
    mask[ys, xs] = ok >= min(need, len(nbrs[i]))
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="real_length / model_length")
    ap.add_argument("--model", default="large", choices=["small", "base", "large"])
    ap.add_argument("--res", type=int, default=770, help="depth network input size")
    ap.add_argument("--max-side", type=int, default=1800)
    ap.add_argument("--tol", type=float, default=0.04, help="consistency tolerance (relative depth)")
    ap.add_argument("--need", type=int, default=2, help="neighbour photos that must agree")
    ap.add_argument("--k", type=int, default=5, help="neighbours to check")
    ap.add_argument("--multi-camera", action="store_true", help="photos from different lenses/phones")
    args = ap.parse_args()

    out = Path(args.out)
    work = out / "work"
    dbg = work / "depth_vis"
    dbg.mkdir(parents=True, exist_ok=True)

    if prep(Path(args.photos), work / "images", args.max_side) < 4:
        raise SystemExit("Need at least 4 sharp photos")
    rec, imgdir = run_sfm(work / "images", work, not args.multi_camera)

    views = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        M = pose(im)
        uv, z = [], []
        for p in im.points2D:
            if p.has_point3D():
                zc = (M[:3, :3] @ rec.points3D[p.point3D_id].xyz + M[:3, 3])[2]
                if zc > 0:
                    uv.append(p.xy)
                    z.append(zc)
        views.append((im.name, M, cam.calibration_matrix(), cam.width, cam.height,
                      np.array(uv), np.array(z)))
    zs = np.concatenate([v[6] for v in views])
    voxel = float(np.median(zs)) / 600
    dmax = float(np.percentile(zs, 95)) * 1.5

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mid = f"depth-anything/Depth-Anything-V2-{args.model.capitalize()}-hf"
    proc = AutoImageProcessor.from_pretrained(mid)
    net = AutoModelForDepthEstimation.from_pretrained(mid).to(dev).eval()

    # Pass 1: depth per photo
    depths = []
    for name, M, K, w, h, uv, z in views:
        if len(z) < 30:
            print("skip (too few SfM points):", name)
            depths.append(None)
            continue
        rgb = cv2.cvtColor(cv2.imread(str(imgdir / name)), cv2.COLOR_BGR2RGB)
        inv = align(predict_disp(rgb, proc, net, dev, args.res), uv, z)
        d = np.where(inv > 1e-4, 1.0 / np.maximum(inv, 1e-4), 0).astype(np.float32)
        gy, gx = np.gradient(d)
        edge = cv2.dilate((np.hypot(gx, gy) / np.maximum(d, 1e-6) > 0.03).astype(np.uint8),
                          np.ones((3, 3), np.uint8)) > 0
        d[edge | (d > dmax)] = 0
        depths.append(d)
        vis = np.clip(inv / max(np.percentile(inv, 99), 1e-6), 0, 1)
        cv2.imwrite(str(dbg / f"{Path(name).stem}.png"),
                    cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
        print("depth:", name)

    # Neighbours = nearby cameras looking the same way
    C = np.array([-v[1][:3, :3].T @ v[1][:3, 3] for v in views])
    V = np.array([v[1][2, :3] for v in views])
    nbrs = []
    for i in range(len(views)):
        order = np.argsort(np.linalg.norm(C - C[i], axis=1))[1:]
        nbrs.append([j for j in order if depths[j] is not None and V[i] @ V[j] > 0.5][:args.k])

    # Pass 2: filter + fuse
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i, (name, M, K, w, h, _, _) in enumerate(views):
        if depths[i] is None:
            continue
        keep = consistency(i, depths, views, nbrs, args.tol, args.need)
        frac = keep.sum() / max(1, (depths[i] > 0).sum())
        print(f"fuse {name}: kept {frac:.0%} of pixels" + ("  <-- views disagree" if frac < 0.3 else ""))
        d = np.where(keep, depths[i], 0).astype(np.float32)
        rgb = cv2.cvtColor(cv2.imread(str(imgdir / name)), cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=dmax, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, intr, M)

    mesh = vol.extract_triangle_mesh()
    cl, n, _ = mesh.cluster_connected_triangles()
    cl, n = np.asarray(cl), np.asarray(n)
    mesh.remove_triangles_by_mask(n[cl] < 0.05 * n.max())
    mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()
    pcd = vol.extract_point_cloud()
    if args.scale != 1.0:
        mesh.scale(args.scale, center=(0, 0, 0))
        pcd.scale(args.scale, center=(0, 0, 0))
    o3d.io.write_triangle_mesh(str(out / "scene_mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(out / "scene_points.ply"), pcd)
    print("done ->", out / "scene_mesh.ply", "(open in MeshLab or Blender)")


if __name__ == "__main__":
    main()
