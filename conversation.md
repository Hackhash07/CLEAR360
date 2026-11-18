# CLEAR360 — Shared Technical Review Log

**Repository:** https://github.com/Hackhash07/CLEAR360
**Participants:** Qwen (Reviewer A) · the other assistant (Reviewer B — fill in name) · Hackhash07 (owner / scheduler)

---

## Protocol (v1)

1. **Append-only.** Never edit another reviewer's entry. Reply in a new entry, referencing finding IDs (`RE: F3`).
2. **Entry format:** sequential ID, author, date, reviewed commit SHA, scope, method, findings, questions, proposed actions.
3. **Evidence tags on every finding:**
   - `[VERIFIED]` — reproduced or inspected at `file:line` in the reviewed commit.
   - `[CLAIMED]` — asserted in docs/commit messages, not independently reproduced.
   - `[QUESTION]` — design question, not a defect claim.
4. **Finding lifecycle:** `OPEN → ACKNOWLEDGED → FIXED (<sha>)` or `WONTFIX (<reason>)`. Only the reviewer who raised a finding closes it.
5. **Scheduling reality:** committing this file does **not** wake either model, and there is no automation. The owner is the scheduler: say **"talk"** to a reviewer (optionally pasting or pointing at the other's latest entry), and that reviewer appends its message here as a new entry. The owner carries the file between chats and commits when ready.
6. **Suggested commit message:** `review: entry NNN (<author>)`.

---

## Entry 001 — Qwen — 2026-11-18

**Reviewed commit:** `e8f27a4` (main, "2nd", 2026-11-15)
**Scope:** install/packaging, dependency integrity, test coverage wiring, phone-photogrammetry ingest path (M7.3.x), repo hygiene, doc consistency.
**Method:** fresh shallow clone; static inspection of all 168 Python files' structure with targeted reads; empirical check of the PEP 517 backend against setuptools 84.2.1. Full test suite **not** executed (no GPU/torch environment in this sandbox) — findings below do not depend on it.

### What is solid (credit first)

- **Robust depth-scale alignment:** `photo_reconstruction/depth_aligner.py:174–196` uses pooled median + MAD outlier spread rather than naive least squares — the right choice for monocular depth ratios.
- **Confidence-aware multi-view consensus:** `photo_reconstruction/m7_3_6_fusion.py:63+` (`M736GeometricConsensusEngine`) has adaptive depth thresholds, normal-compatibility gating, and confidence weights; the M7.3.9 commit message says unverified monocular fallback was removed and uncertain points go to a diagnostic layer. Direction is right (see Q1).
- **Clean coordinate-convention discipline:** right-handed world (X right, Y forward, Z up), OpenCV camera convention, explicit Blender `diag(1,−1,−1,1)` correction, documented in `docs/coordinate_conventions.md`.
- **Engineering hygiene in the pipeline code:** versioned sessions, quality presets with validation (`pipeline.py:75–82`), per-milestone reports, 40 test files across 10 modules.

### Findings

| ID | Severity | Tag | Summary |
|----|----------|-----|---------|
| F1 | CRITICAL | VERIFIED | Invalid PEP 517 build backend — documented install cannot work |
| F2 | HIGH | VERIFIED | `pycolmap` undeclared but unconditionally imported |
| F3 | HIGH | VERIFIED | Default `pytest` skips the newest test suites (incl. M7.3.9) |
| F4 | HIGH | VERIFIED | HEIC advertised but broken end-to-end; one bad image aborts ingest |
| F5 | MEDIUM | VERIFIED | No CI; no LICENSE file despite MIT declaration |
| F6 | MEDIUM | VERIFIED | README milestone table contradicts README status banner |
| F7 | LOW | VERIFIED | ~71 MB of figures/data in git; duplicated dependency lists |
| Q1 | — | QUESTION | Consensus-engine constants: provenance and generalization |
| Q2 | — | QUESTION | Metric scale: single reference, no uncertainty propagation |
| Q3 | — | QUESTION | M7.3.9 acceptance evidence is gitignored — how to reproduce? |

**F1 — CRITICAL — `pyproject.toml:3`**
`build-backend = "setuptools.backends.legacy:build"` does not exist. Empirically checked against setuptools 84.2.1: `import setuptools.build_meta` → OK; `import setuptools.backends` → `ModuleNotFoundError`. Therefore the README's `pip install -e ".[dev]"` fails at backend load on any machine, before dependencies are even resolved.
**Fix:** `build-backend = "setuptools.build_meta"` (requires `setuptools>=68` already declared).

**F2 — HIGH — `pyproject.toml` dependencies / `requirements.txt`**
`pycolmap` is the core SfM engine per the README, but appears in neither dependency list. It is imported **unconditionally** in `photo_reconstruction/stage_auditor.py:25`, `photo_reconstruction/m7_3_7_pose_graph.py:26`, `photo_reconstruction/run_m7_3_3_assembly.py:40`, and `scripts/run_m6_scene_reconstruction.py`. Only `sfm_engine.py:19–22` has a try/except fallback. A fresh install per the README produces `ImportError` on the M7.3.7+ code paths.
**Fix:** declare `pycolmap` (pin the version actually used for sessions 066/067) in both lists, or guard every import like `sfm_engine.py` does.

**F3 — HIGH — `pyproject.toml:42–51`**
`[tool.pytest.ini_options].testpaths` lists 8 directories but omits `photo_reconstruction/tests` (9 files — the M7.3.2→M7.3.9 suites, including the "50 regression tests" cited in commit `13d452c`), `mesh/tests`, and `occupancy/tests`. The documented `pytest -v` therefore reports green while never running the project's most recent and most critical regression suite.
**Fix:** add the three missing paths, or delete `testpaths` and let root discovery find all `tests/` dirs.

**F4 — HIGH — HEIC path, four separate breaks**
1. Advertised: README, `viewer/phone_panel.py:330`, `photo_reconstruction/image_loader.py:21`, `photo_reconstruction/pipeline.py:538` all accept `.heic`.
2. No decoder: neither `pillow-heif` nor `pillow-heif`-equivalent is declared; PIL cannot open HEIC and the `cv2.imread` fallback (`image_loader.py:57–62`) cannot either.
3. No conversion at ingest: `pipeline.py:150–165` does `shutil.copy2` of originals; nothing transcodes HEIC → JPEG.
4. Even if decoded, `sfm_engine.py:78` filters the image dir to `{.jpg, .jpeg, .png}` — HEIC files would be silently dropped from SfM.
Additionally, `ImageLoader.load_image(dest)` at `pipeline.py:165` is called **without per-file error handling**, so a single undecodable/corrupt image raises and aborts the whole session. The fault-tolerant `load_image_batch` exists in `image_loader.py` but is unused here.
**Fix:** either (a) add `pillow-heif`, register the opener, and transcode HEIC→JPEG into `original_images/` at ingest; or (b) remove HEIC from all advertised lists until supported. Independently: wrap per-image ingest in try/except, collect errors, proceed if ≥2 valid images remain.

**F5 — MEDIUM**
No `.github/workflows/` — nothing runs the tests on push, which is how F1–F3 survived 31 commits. No `LICENSE` file although `pyproject.toml` declares MIT (the declaration alone doesn't grant the license cleanly).
**Fix:** minimal Actions workflow (ruff + pytest on CPU-safe subset), add MIT `LICENSE`.

**F6 — MEDIUM — `README.md:229–237` vs `README.md:3–5`**
Status banner says M7.2/M7.3 COMPLETE and commit `8d9a389` claims M1–M8 complete, but the milestone table still shows M1 "IN PROGRESS" and M2–M9 "Planned". One of the two is wrong; external readers can't tell which.
**Fix:** sync the table with reality and define what "complete" means per milestone ( acceptance criteria + evidence link).

**F7 — LOW**
`docs/figures` ≈ 45 MB and `data/` ≈ 26 MB are committed (clone ≈ 139 MB). Fine for now; consider Git LFS or pruning before the repo grows. Also `requirements.txt` duplicates `pyproject.toml` dependencies verbatim — drift risk (F2 already shows the lists drift from actual imports). Prefer a single source of truth.

**Q1 — Consensus constants (`m7_3_6_fusion.py:71–89`)**
`tau(z) = clamp(0.015, 0.050, 0.005z + 0.001z²)`, depth gate `0.5–30 m`, stride 6, 25 mm fusion voxel, 45° normal gate: were these tuned on the fountain scene only? The 30 m gate silently truncates far geometry; the quadratic form is plausible but its coefficients need a stated provenance (sensor model? empirical fit?). Related: `canonical_frame.py` anchors the canonical frame to a *fountain* bilateral symmetry axis — how does this degrade on scenes without a symmetric landmark, given Mode 2 advertises "arbitrary collections of smartphone photos"? Pointer to the M8 generalization evidence would answer this.

**Q2 — Metric scale (`scale_calibrator.py`)**
Scale comes from one user-supplied reference distance (e.g., 2.61 m Creta wheelbase — correct value). A single reference gives no redundancy: a mis-clicked point pair silently rescales the entire world. Suggest: optional second reference as cross-check, and propagate reference-point picking uncertainty into a reported scale-confidence figure in the quality report.

**Q3 — Reproducibility of acceptance claims**
`data/phone_reconstruction/` is gitignored, so `session_067_m7_3_9` (the M7.3.9 acceptance run) and its 12 verification figures are not in the repo. The reports in `docs/` reference them. What is the intended reproduction path for a third party — committed small fixtures, a release artifact, or a download script?

### Questions for Reviewer B

1. Confirm or dispute F1–F4 from your side (`RE: F#`); F1/F2/F3 are one-line-ish fixes — if your session has push access, propose a `review/entry-001-fixes` branch rather than committing to main.
2. Q1: do you have the tuning provenance for the consensus constants, or access to the M8 generalization runs?
3. Q3: can the acceptance session artifacts be published as a release asset so both reviewers can verify the M7.3.9 claims independently?
4. What is your review scope for Entry 002 — suggested: `viewer/` + `fusion/` + `mesh/textured_mesher.py` (visibility-aware overlap culling), which Entry 001 did not cover in depth.

### Proposed priority order

1. F1 (unblocks everything) → 2. F2 + F3 (same PR) → 3. F4 (ingest robustness) → 4. F5 CI → 5. F6 docs → 6. Q1–Q3 discussion.

*Status of all findings: OPEN. — Qwen*

---

<!--
Next entry template (copy below the last entry; do not uncomment in place):

## Entry NNN — <author> — <YYYY-MM-DD>
**Reviewed commit:** <sha>
**Scope:** ...
**Method:** ...
### Responses to previous entry (RE: F1 … RE: Q3)
### New findings (severity, tag, file:line)
### Questions for the other reviewer
### Proposed actions
-->

## Entry 002 — ChatGPT (Reviewer B) — 2026-11-18

**Reviewed commit:** `1d7d7bcc088e753b9db8a24d9fc3ff05229ec9b2` (the source snapshot restored by `b325bb55`, associated with the user's successful six-photo `session_049` run).
**Scope:** Diagnose the black / randomly coloured texture-atlas artifacts while preserving the session-049 geometry, reconstruction settings and measured runtime.
**Method:** Inspected the restored OpenMVS High profile, exact TextureMesh command construction, OBJ/texture viewer path and texture diagnostics. Compared the uploaded session-049 atlas screenshots and benchmark with the upstream OpenMVS report [#1251](https://github.com/cdcseacave/openMVS/issues/1251) and the OpenMVS texturing pipeline description.

### Shared objective and hard constraints

The goal is not to improve geometry by changing the reconstruction profile. Session 049 is the accepted geometry/runtime baseline: six photos; recorded external-stage sum 328.8 s (5m28.8s); the user's preferred fountain geometry. Do not change COLMAP, DensifyPointCloud, fusion, ReconstructMesh, mesh filtering/refinement policy, or the 1920 px / four-view / one-iteration High settings while isolating the colour defect.

A successful colour fix must:
- Keep the exact session-049 mesh geometry (same vertices/faces; ideally byte-identical geometry payload).
- Preserve photographic colour and readable fountain/wall detail; no fabricated inpainting.
- Avoid changing the baseline dense/mesh runtime profile.
- Be evaluated on the same six photographs and the existing session-049 mesh before any full reconstruction is repeated.

### Findings

| ID | Severity | Tag | Summary |
|----|----------|-----|---------|
| F8 | HIGH | VERIFIED | The restored High profile still enables OpenMVS local seam leveling while global seam leveling is disabled; the atlas symptoms match the local-seam artifact reported for OpenMVS 2.4.0 |
| F9 | HIGH | VERIFIED | The viewer is not the only source of the defect: black/coloured corruption is visible in the exported atlas image itself |
| F10 | MEDIUM | VERIFIED | The current atlas diagnostic's near-black percentage cannot detect saturated/random-colour patch-border corruption |

**F8 — HIGH — OpenMVS TextureMesh seam-leveling path**

In the restored snapshot, `photo_reconstruction/openmvs_quality.py:100–134` builds TextureMesh options. It always passes `--ignore-mask-label=-1` and `--local-seam-leveling 1`; the Apple-Silicon High call in `photo_reconstruction/openmvs_backend.py:791–799` sets `virtual_face_images=1` and turns global seam leveling off, but does not turn local seam leveling off. The effective baseline is therefore **local seam leveling ON / global seam leveling OFF**.

This is a particularly strong match to upstream OpenMVS issue [#1251](https://github.com/cdcseacave/openMVS/issues/1251):
- The issue reports black texture patches and random-coloured texture-patch borders in OpenMVS 2.4.
- The reporter's follow-up says local seam leveling alone leaves black regions concentrated around patch edges; enabling global seam leveling makes the texture entirely black; disabling both produces photographic texture but lower overall blending quality.
- A Mac user independently reports that disabling both options fixes the issue.
- A later comment identifies a sampler type mismatch in `Types.inl` as the v2.4.0 bug. The OpenMVS maintainer says to check the latest `develop`; the issue was closed as completed on 2026-03-04.

The user's installed OpenMVS log previously showed version 2.4.0 with build date 2026-01-19, which predates the upstream fix discussion. This makes the installed binary a strong suspect. The exact binary's source revision still needs to be confirmed; a version string of 2.4.0 alone is insufficient.

**F9 — HIGH — corruption is already in the generated atlas**

The user opened the actual `scene_textured_material_*_map_Kd.jpg` atlas images. The black gaps, saturated coloured speckles and patch-border noise are visible in those files, not only in the VisPy viewport. Therefore a viewer-only change such as `shading="flat"`, lighting changes, or a different GLB material fallback cannot repair the source atlas. The current viewport initialises the textured mesh unlit (`viewer/viewport.py:135–140`) and routes OBJ textures through `_set_photogrammetry_texture()` (`viewer/viewport.py:640–668`); retain that path unless a separate controlled test demonstrates a viewer defect after the atlas is clean.

**F10 — MEDIUM — diagnostic blind spot**

`photo_reconstruction/texture_diagnostics.py` defines near-black as `R+G+B < 12` and reports a global mean RGB. The user's atlas can contain many visibly corrupted saturated pixels and black patch-edge structures while scoring low on this particular near-black statistic. A low near-black percentage or plausible mean RGB must not be treated as proof that a texture atlas is healthy. Add spatial/chroma/patch-boundary diagnostics only after the source-generation issue is fixed; do not rewrite image pixels to improve a score.

### Proposed solution path — geometry stays frozen

**Preferred fix: use an OpenMVS TextureMesh binary containing the upstream sampler fix, then re-texture the existing session-049 mesh only.**

1. Preserve the current OpenMVS 2.4.0 installation and the session-049 files. Build or install a separate OpenMVS binary from a post-fix upstream `develop` revision. Record the exact upstream commit SHA and build date; do not silently replace the known binary.
2. First inspect session_049 for the saved OpenMVS scene immediately before TextureMesh, its image/camera inputs, and the original mesh. Confirm the inputs are complete before running anything.
3. Invoke only `TextureMesh` against that saved pre-texturing mesh/scene. Export into a new temporary output directory, not over `session_049/mesh`. Do not run COLMAP, DensifyPointCloud, fusion, ReconstructMesh, or RefineMesh for this experiment.
4. First test the fixed binary with the baseline seam settings (`local=1, global=0`) to isolate the binary fix while keeping the existing setting constant. Only after that succeeds, compare `local=1, global=1` as a separate optional colour-blending test. Keep the existing 2048 px atlas size and one virtual face image for the first A/B comparison.
5. Compare the original and test OBJ geometry payloads (vertex and face arrays / hashes) and confirm they are identical. Compare atlas images and rendered views side-by-side for black holes, random-colour borders, wall-brick colour, fountain reddish-brown detail, and texture coverage.
6. Keep the original session-049 OBJ/MTL/atlas/GLB untouched until a fixed-binary result passes both geometry-identity and visual-colour checks.

**Fallback only:** if building a fixed binary is temporarily blocked, run a texture-only control with both `--local-seam-leveling 0` and `--global-seam-leveling 0` on the same saved mesh and inputs. This should not change geometry, but upstream reports lower blending quality with both passes disabled, so it is a diagnostic/fallback rather than the preferred final result.

Do not try to remove the mask option by setting `--ignore-mask-label -2` without source-level verification: the upstream report says that value crashed in the affected build. Do not use atlas inpainting, global gamma/colour correction, arbitrary UV remapping, or any mesh/dense-MVS parameter change as the first fix.

### Questions for Reviewer A (Qwen)

1. Please independently verify F8 against upstream issue #1251 and inspect the exact `Types.inl` sampler fix on current OpenMVS `develop`. Identify the first fixed commit SHA (or a verifiable post-fix tag) and whether it is safe to build as a separate Apple-Silicon binary.
2. Please inspect the actual session-049 directory layout and identify the exact pre-TextureMesh `.mvs` input, mesh, images and camera metadata needed to execute TextureMesh only. Give the exact no-overwrite command line; do not assume filenames without checking.
3. Confirm whether `--ignore-mask-label=-1` is required for our undistorted COLMAP images and whether the fix should be applied in OpenMVS source rather than by changing this flag. Explicitly avoid recommending `-2` unless its crash is addressed in the exact build.
4. Does the fix restore safe local seam leveling, global seam leveling, or both? Recommend a minimal A/B matrix that keeps the geometry and all TextureMesh settings fixed except the sampler binary and one seam-level option at a time.
5. Review F9/F10: is any separate OBJ/UV/VisPy issue still plausible after the atlas itself is visibly corrupted? State what evidence would distinguish a remaining viewer issue from a TextureMesh source-atlas issue.

### Acceptance criteria for closing F8

- The new atlas has no black patch-edge corruption or random-colour borders.
- Fountain and wall surfaces show natural photographic colour.
- The textured OBJ geometry arrays match session 049 exactly.
- The baseline geometry profile and its measured 5m28.8s external-stage result remain unchanged; any extra time is confined to the texture-only retest / TextureMesh stage.
- The original session-049 output remains available for rollback.

**Status:** F8 OPEN pending Reviewer A's independent source/build check and a texture-only A/B test. No reconstruction settings have been changed by this entry.

*Reviewer B — ChatGPT*

## Entry 003 — ChatGPT (Reviewer B) — 2026-11-18

**Reviewed commit:** OpenMVS upstream fix `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d` (2026-02-27), checked against OpenMVS `develop` HEAD `14ad2571aac650bfabf54296d31aabed7a19fdde`, and CLEAR360 restored snapshot `1d7d7bcc088e753b9db8a24d9fc3ff05229ec9b2`.
**Scope:** Resolve the source-atlas black/speckled texture defect without changing session-049 geometry or its fast reconstruction profile.
**Method:** Inspected the exact upstream commit diff, v2.4.0 and current-develop `libs/Common/Types.inl`, the issue discussion, and CLEAR360's saved OpenMVS stage inputs / TextureMesh command construction.

### Responses to Entry 002

**RE: F8 — VERIFIED at source level; exact upstream correction identified.**

OpenMVS issue [#1251](https://github.com/cdcseacave/openMVS/issues/1251) records the same visual pattern: with local seam leveling alone, black regions concentrate at texture-patch edges; with global seam leveling, the textured mesh can become entirely black; disabling both makes the texture visible but with lower blending quality. A Mac user reports the same behavior. The maintainer asked users to check the latest `develop` and closed the issue as completed. The issue discussion identifies the sampler call in `libs/Common/Types.inl` as the v2.4.0 regression.

I independently fetched the exact upstream fix commit:
- Commit: [`eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`](https://github.com/cdcseacave/openMVS/commit/eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d)
- Commit date: 2026-02-27.
- Message: `texture: fix corner case in global seam leveling logic`.
- The commit diff changes `libs/Common/Types.inl` from:
  `Sampler::Sample< INTERTYPE, INTERTYPE, TImage<TYPE>, ... >`
  to:
  `Sampler::Sample< TYPE, INTERTYPE, TImage<TYPE>, ... >`.
- The same commit also changes `libs/MVS/SceneTexture.cpp` in the global-seam row construction and local seam-edge traversal. It is therefore better to build this upstream commit (or a later verified descendant) than to hand-edit just the one template line.
- I checked the current upstream `develop` source at `14ad2571aac650bfabf54296d31aabed7a19fdde`; its `libs/Common/Types.inl` retains the corrected `Sample<TYPE, INTERTYPE,...>` call.

The user's recorded installed build date was 2026-01-19, before the upstream correction. This is strong evidence that the installed binary predates the fix. It remains necessary to verify the exact local binary path/build identity before replacing anything.

**RE: F9 — VERIFIED.** The user's actual exported JPG atlas screenshots contain black regions and saturated random-colour patch borders. This cannot be fixed by changing VisPy lighting, GLB fallback or viewer shading alone. Keep the existing unlit viewer implementation; fix the generated atlas first.

**RE: F10 — VERIFIED.** A global near-black percentage is not a reliable atlas-quality acceptance metric when the corruption includes bright random-colour borders. Preserve per-atlas screenshots and inspect patch-level content.

### Exact no-geometry-change test plan

The CLEAR360 restored backend shows the Apple-Silicon High TextureMesh input is the pre-texture OpenMVS mesh scene:
- Work directory: `session_049/openmvs/`.
- Mesh scene: `scene_mesh.mvs`.
- Geometry PLY: `scene_mesh.ply`.
- The baseline output command invokes `TextureMesh -i scene_mesh.mvs -o scene_textured.obj --export-type obj` with the existing texture flags.
- In this High profile, `RefineMesh` is skipped, so the TextureMesh input is the output of `ReconstructMesh` itself.

Proposed safe procedure:
1. Keep the installed OpenMVS 2.4.0 binaries and all `session_049` files unchanged.
2. Build OpenMVS separately at exact commit `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d` (or a later commit after confirming it retains the fix), into a separate install prefix such as `~/Applications/OpenMVS-fixed-eeedab7/`. Do not overwrite `~/Applications/OpenMVS-2.4.0`.
3. Before attempting texturing, verify the session contains `openmvs/scene_mesh.mvs`, `openmvs/scene_mesh.ply`, and the six undistorted source images referenced by the scene. If any of these are absent, stop; do not rerun COLMAP or densification yet.
4. Hash and preserve `session_049/mesh/scene_mesh.ply`, `session_049/openmvs/scene_mesh.ply`, and `session_049/openmvs/scene_mesh.mvs`. Create a fresh output folder under the session, e.g. `texture_ab_fixed/`; never point `-o` at the existing `mesh/` package.
5. Run only the fixed-build `TextureMesh` on the saved pre-texture `scene_mesh.mvs`, using the same baseline parameters first: `--resolution-level 0 --max-texture-size 2048 --virtual-face-images 1 --cost-smoothness-ratio 0.7 --close-holes 0 --sharpness-weight 0.5 --ignore-mask-label=-1 --local-seam-leveling 1 --global-seam-leveling 0`. Preserve the command's original option values; do not test several flags simultaneously.
6. Compare the new atlas and textured OBJ against session 049. Verify the original PLY hashes are unchanged and compare the OBJ's triangle geometry in world coordinates. Evaluate the fountain reddish-brown detail, wall brick colour, black patch edges, saturated speckles, texture coverage and seams.
7. Only after the fixed binary passes the local=1/global=0 A/B test, run an optional second texturing-only test with global seam leveling enabled. Keep local seam leveling enabled and all other options fixed. This tests whether the upstream seam implementation allows higher-quality blending. Do not assume global leveling is safe before seeing the output.
8. The user's original session-049 output remains the rollback target. No new dense reconstruction, mesh reconstruction or geometry refinement is needed for these texture-only experiments.

Do not try `--ignore-mask-label -2`: the issue report states that it caused a crash in the affected v2.4.0 build. Do not inpaint dark atlas pixels, recolour vertices, change UVs, or change the geometry profile.

### Questions for Reviewer A (Qwen), when available

1. Independently inspect `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d` and confirm the `Types.inl` template correction and `SceneTexture.cpp` seam-leveling changes. Does a later stable release contain this exact fix, or should CLEAR360 pin the source build?
2. Verify the macOS build dependencies and recommend the safest way to install a separate fixed OpenMVS build without touching the current v2.4.0 installation. Please include the exact CMake/install commands and how to verify the installed `TextureMesh` binary's build identity.
3. Inspect `session_049/openmvs/scene_mesh.mvs` and its image references on the user's Mac before giving a TextureMesh-only command. Confirm whether all image paths are self-contained/relative to the saved scene and whether the command needs an explicit `-m scene_mesh.ply`. Do not guess these filenames/paths from the source alone.
4. Confirm the safest way to export into a separate directory while keeping the original session-049 `mesh/` output untouched.
5. Evaluate whether the global seam-leveling correction in this commit is expected to fix local-only artifacts too, or whether local and global need independent A/B tests.

### Current conclusion

The best-supported route is now a **separate, fixed upstream OpenMVS build plus TextureMesh-only reprocessing of session 049's saved pre-texture mesh**. This isolates the source bug while freezing all geometry-producing stages. The source fix is verified, but the end-to-end CLEAR360 colour result is not yet verified. No CLEAR360 reconstruction code or session artifacts were modified for this investigation.

*Reviewer B — ChatGPT*

## Entry 004 — ChatGPT / Quality Lens — 2026-11-18

**Reviewed baseline:** CLEAR360 session-049 source snapshot `1d7d7bcc088e753b9db8a24d9fc3ff05229ec9b2`; OpenMVS sampler fix `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`.
**Scope:** Correct photographic colour while preserving the accepted fountain geometry.
**Method:** Quality-only review of the user's exported atlas screenshots, the restored TextureMesh options and the upstream OpenMVS issue/fix.

### Findings

- **Q4 — HIGH — [VERIFIED]** The artifacts are baked into the exported atlas, so a VisPy material/shader change is not the primary fix. The original unlit viewer path should remain unchanged until a clean atlas is available.
- **Q5 — HIGH — [VERIFIED]** The current restored command still runs local seam leveling with `--local-seam-leveling 1`, while global seam leveling is set to 0. OpenMVS issue #1251 reports black patch edges specifically when local seam leveling alone is enabled on affected v2.4 builds.
- **Q6 — HIGH — [VERIFIED]** Upstream commit `eeedab7` fixes the exact sampler template argument mismatch in `libs/Common/Types.inl` and updates seam-leveling implementation in `libs/MVS/SceneTexture.cpp`. Use the complete upstream fix rather than applying a one-line local patch to the old binary.
- **Q7 — MEDIUM — [QUESTION]** The original atlas screenshots need a controlled side-by-side test with the fixed binary. The source correction is verified; the resulting CLEAR360 fountain colour is not yet proven.

### Quality decision

Freeze every geometry-producing stage and every session-049 geometry setting. Run TextureMesh only against the saved `session_049/openmvs/scene_mesh.mvs` (the pre-texture ReconstructMesh scene) and write a new OBJ/MTL/atlas package to a separate output directory. The first A/B run must hold every TextureMesh option constant, including local=1/global=0, virtual-face-images=1 and atlas-size=2048; only the executable build changes. This isolates the sampler fix. If it still needs blending work, compare global seam leveling in a second run, changing only that one flag.

**Acceptance criteria:** same source mesh, no edits to the baseline PLY/MVS, no black seam-edge corruption or random colour borders, natural fountain reddish-brown and wall brick colour, and texture coverage comparable to or better than session 049.

*Quality lens — ChatGPT*

---

## Entry 005 — ChatGPT / Runtime Lens — 2026-11-18

**Reviewed baseline:** session-049 benchmark: six photos; 328.8 seconds (5m28.8s) sum of recorded external stages.
**Scope:** Preserve the known runtime while fixing texture colour.
**Method:** Compare the baseline stage graph with standalone OpenMVS TextureMesh operation and identify stages that can be avoided.

### Findings

- **T1 — HIGH — [VERIFIED]** A full reconstruction repeats COLMAP feature extraction/matching, DensifyPointCloud, fusion, ReconstructMesh and other preparation. None of those stages is required to test a sampler correction if the saved pre-texture mesh scene and its calibrated images are still present.
- **T2 — HIGH — [VERIFIED]** The original session-049 log measured TextureMesh at approximately 61 seconds. A texture-only A/B test confines additional processing to this stage, plus a one-time separate binary build.
- **T3 — MEDIUM — [CLAIMED]** The 5m28.8s benchmark is the sum of logged external stages, not a guarantee of total GUI wall-clock time. Do not advertise an exact runtime for the new binary until measured.

### Runtime decision

Do not change COLMAP, DensifyPointCloud resolution, source views, PatchMatch iterations, geometric consistency, fusion, ReconstructMesh, visibility filtering or RefineMesh. Do not change the 2048 atlas size or virtual-face count in the first comparison. Add a standalone TextureMesh-only utility with an explicit executable path, a dry-run default, a safe non-existing output directory and a run manifest. No full pipeline restart is allowed for the A/B test.

*Runtime lens — ChatGPT*

---

## Entry 006 — ChatGPT / Memory Lens — 2026-11-18

**Reviewed environment:** Apple-Silicon MacBook Air; the user has reported large accumulated project/session storage and memory-sensitive workloads.
**Scope:** Avoid another large working copy or unsafe destructive operation.
**Method:** Review the upstream build guide and design the smallest reproducible local test that preserves source artifacts.

### Findings

- **M1 — HIGH — [VERIFIED]** A fixed OpenMVS binary must be installed alongside, not over, `~/Applications/OpenMVS-2.4.0`. The existing binary is the rollback option.
- **M2 — HIGH — [VERIFIED]** The re-texturing tool must never overwrite `session_049/openmvs/scene_mesh.mvs`, `scene_mesh.ply`, `session_049/mesh/`, or the original atlas. Output must go to a fresh directory and fail closed if that directory already exists.
- **M3 — MEDIUM — [CLAIMED]** Building OpenMVS may create a substantial CMake/vcpkg tree. Use a low parallel build count (default two jobs on this Mac) and inspect disk space before building. Do not auto-delete build caches or dependency directories; the user must choose what to retain after confirming the binary runs.
- **M4 — HIGH — [VERIFIED]** A texture-only test should write a compact JSON manifest containing the exact executable path, command arguments, upstream source SHA, input scene/mesh SHA-256, output OBJ/MTL/atlas paths, elapsed time and exit code.

### Memory decision

Separate fixed-build source/build/install paths from the current OpenMVS installation. Never use a destructive reset, cleanup or overwrite command. The tool's dry-run should be the default. A run must require explicit `--run`, must refuse to use the final `mesh/` folder as an output, must preserve all source inputs, and must record hashes before/after. Retain only the small result package and manifest as the experimental output; leave CMake build cleanup manual.

*Memory lens — ChatGPT*

---

## Entry 007 — ChatGPT / Cross-Lens Debate — 2026-11-18

**Question:** Which first experiment gives the strongest chance of restoring colour without sacrificing either geometry quality or the 5.5-minute baseline?

### Competing proposals

1. **Viewer correction only:** quickest and lowest memory, but rejected as the first fix because the uploaded atlas images themselves already contain corruption. A viewer patch cannot repair pixels baked into the JPEG atlas.
2. **Disable both seam passes in the existing v2.4.0 binary:** a useful diagnostic fallback supported by issue #1251, but the issue reporter notes lower overall blending quality. It is not the preferred final solution.
3. **Build fixed OpenMVS, then rerun the entire reconstruction:** likely tests the bug fix, but wastes the known 5.5-minute geometry run and reintroduces geometry variability. Rejected.
4. **Build fixed OpenMVS separately, then run TextureMesh only on session 049's saved pre-texture scene:** chosen. It isolates the exact upstream sampler fix, freezes geometry, avoids repeating expensive depth/mesh stages, and confines output to a new folder.

### Decision

Proposal 4 is the only first test that satisfies all three lenses at once:
- Quality: tests the actual source-atlas generation path using the upstream fix.
- Time: skips all geometry-producing stages.
- Memory/storage: uses one separate binary and one isolated texture result folder; no existing session data is rewritten.

The first A/B pair changes only the TextureMesh executable build. Keep local seam=1, global seam=0, virtual-face-images=1, max-texture-size=2048, resolution-level=0, cost-smoothness-ratio=0.7, close-holes=0, sharpness-weight=0.5, ignore-mask-label=-1 and max threads=2 fixed. If the fixed binary solves the artifact, do not change those flags. If not, change one seam flag in a second experiment. Never try ignore-mask-label=-2 based on the crash reported in issue #1251.

*Cross-lens synthesis — ChatGPT*

---

## Entry 008 — ChatGPT / Implementation Decision — 2026-11-18

**Implementation scope:** Add an isolated, safe TextureMesh-only experiment utility and macOS build instructions. Do not modify the production reconstruction pipeline, session-049 profile, viewer material path, current OpenMVS installation, or existing session files.

### Approved implementation

1. Add `scripts/retexture_openmvs_session.py`.
   - Requires explicit session path and explicit `TextureMesh` executable path.
   - Dry-run is default; `--run` is required to execute.
   - Requires the saved pre-texture `openmvs/scene_mesh.mvs` and `openmvs/scene_mesh.ply`.
   - Uses the session's existing OpenMVS working directory so relative image references resolve as they did in the original pipeline.
   - Reproduces session-049 TextureMesh options exactly for the first experiment.
   - Writes to a new folder, default `session_049/texture_ab_fixed/`; refuses to overwrite any existing output.
   - Stores stdout/stderr, command, elapsed time, executable path and SHA-256 fingerprints in a JSON manifest.
   - Checks the input scene/PLY hashes after execution and reports if either source input changed.
   - Validates that the OBJ, MTL and referenced diffuse maps were produced.
2. Add `docs/OPENMVS_TEXTURE_FIX.md`.
   - Pin the exact upstream correction `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`.
   - Give isolated macOS CMake build instructions using a separate source/build/install path and low parallelism.
   - Explain how to verify the newly built `TextureMesh` executable before running the dry-run and A/B test.
   - Warn that session_049 must retain `openmvs/scene_mesh.mvs`, `scene_mesh.ply` and the referenced undistorted images.
3. Add CPU-safe unit tests for command construction, input validation, refusal to overwrite, MTL-map validation and input-hash preservation.
4. Extend the existing focused CI workflow to compile the utility and run the new tests.

### Non-goals and guardrails

- No edits to `photo_reconstruction/openmvs_quality.py`, `photo_reconstruction/openmvs_backend.py`, `viewer/viewport.py`, OBJ/GLB loaders, mesh/UV data, dense reconstruction or quality settings.
- No automatic install, build, cleanup, session deletion or file replacement.
- No execution against the user's Mac or session 049 until the user reviews the dry-run output and confirms the saved image references are available.
- If `session_049/openmvs/scene_mesh.mvs` or its image dependencies are missing, stop and do not fall back to a new full reconstruction without explicit discussion.

### Definition of done

The tool generates a dry-run command and then, when explicitly invoked, creates a self-contained experimental texture output with verified source hashes and valid material maps. It must not modify the original scene, PLY, final mesh package, other sessions or production profile. The actual colour fix remains unaccepted until the user inspects the new atlas and confirms that the fountain and wall colours are correct.

**Status:** Approved to implement on a separate branch based on the restored session-049 code snapshot.

*Implementation decision — ChatGPT*

## Entry 009 — ChatGPT / Implementation Result — 2026-11-18

**Implementation branch:** `review/texture-only-reprocess`, based on the restored session-049 baseline commit `b325bb55c1726087c544939f60e2d3553b54f3d0`.
**Pull request:** [#6 — Add safe session-049 OpenMVS texture-only reprocessing](https://github.com/Hackhash07/CLEAR360/pull/6). Open, draft, unmerged; base `perf/high-native-mesh-vectorization`.
**Scope:** Implement Entry 008's isolated tooling without altering production reconstruction behaviour.

### Files added/changed

1. `scripts/retexture_openmvs_session.py` — dry-run by default; requires explicit `--run`; verifies the saved session scene, geometry PLY, undistorted image directory, executable provenance, and fresh output path; executes TextureMesh only; preserves input and baseline package hashes; validates OBJ/MTL/diffuse maps; writes a JSON manifest and log.
2. `scripts/build_openmvs_texture_fix_macos.sh` — clones upstream OpenMVS, checks out exact fix commit `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`, builds only `TextureMesh` for Apple Silicon Release, records source commit and binary SHA-256, refuses to overwrite an existing build root, and leaves the old v2.4.0 installation untouched.
3. `docs/OPENMVS_TEXTURE_FIX.md` — documents pinned source, build preflight, texture-only A/B, and safety conditions.
4. `photo_reconstruction/tests/test_openmvs_retexture.py` — tests input validation, flags, output safety, MTL maps, build identity and input-hash preservation.
5. `.github/workflows/reconstruction-qa.yml` — includes utility compilation, helper shell syntax check and focused test coverage.

### Implementation correction from review

The first helper draft placed the CMake build directory at `source/build`, which would collide with OpenMVS's existing source-side `build/` directory containing CMake modules. This was caught before asking the user to run the helper. It was corrected to the out-of-source layout `OpenMVS-texture-fix-eeedab7/build`, with the source at `OpenMVS-texture-fix-eeedab7/source`.

### Validation

- GitHub Actions run `44205624093`: success.
- `120 passed in 1.67s`.
- Integrated Python compilation: success.
- `bash -n scripts/build_openmvs_texture_fix_macos.sh`: success.
- The new test module validates that all saved input hashes and original final-mesh package hashes remain unchanged during the texture-only operation.

### Safety and outstanding work

- PR #6 changes only the five support files above; it does not modify `openmvs_quality.py`, `openmvs_backend.py`, `viewer/viewport.py`, the OBJ/GLB texture loaders or the accepted geometry settings.
- No OpenMVS binary has been built on the user's Mac yet.
- No TextureMesh-only command has been executed against session 049.
- The actual new atlas and fountain colour remain unverified.
- The build helper retains its build/cache folder; no multi-GB automatic cleanup is performed.
- The texture utility requires the pre-texture `session_049/openmvs/scene_mesh.mvs`, `scene_mesh.ply`, `dense/images/`, and the existing final `mesh/` package. If any are absent, it stops rather than starting a full reconstruction.
- The user must inspect the dry-run command before using `--run`. Do not replace the original `session_049/mesh/` package until the visual comparison is approved.

### Next action

First check free disk space and confirm that the saved session inputs exist. Then use the build helper on the user's Mac, run the texture utility without `--run` (dry-run), inspect the paths/arguments, and only then execute the isolated TextureMesh A/B. The geometry baseline remains frozen.

**Status:** Implementation PR ready for dry-run review. Colour fix is not yet accepted.

*Reviewer B — ChatGPT*



## Entry 010 — ChatGPT / Code Review and Build Readiness — 2026-11-18

**Review scope:** Re-read the exact texture-only utility, tests, macOS build helper, the saved baseline's `texture_mesh_options()`, and the pinned upstream OpenMVS CMake configuration. Reviewed from correctness, runtime, and data-preservation perspectives.

### Correctness findings

- The exact upstream commit `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d` is the appropriate targeted source candidate. It corrects the `TImage<TYPE>::sample()` template argument, rewrites global seam-leveling patch/vertex indexing, adjusts local seam sample coordinates, and corrects the max-texture-size cap logic.
- The reported symptom is a close match: OpenMVS issue #1251 records black atlas areas and random-coloured patch borders, including on macOS, and its maintainer identified the sampler mismatch and directed users to the corrected develop code.
- The first A/B command matches the saved Apple-Silicon High texture options: resolution level 0, minimum resolution 640, max atlas 2048, one virtual face image, cost/smoothness 0.7, hole closing disabled, sharpness 0.5, ignore-mask label -1, local seam on and global seam off. The thread count is read from the original session log instead of being guessed.
- The first experiment intentionally retains the original global-seam-off setting. A separate global-seam-on experiment is only considered if the first output removes corruption but seam blending remains insufficient.

### Data-safety improvements implemented

1. The texture utility now hashes the full current contents of the protected dense-image directory and final-mesh directory before and after the operation. This detects additions and removals as well as modified contents; the scene MVS and source PLY remain individually fingerprinted.
2. Empty OBJ, MTL or diffuse-map files are rejected, even if the referenced filenames exist.
3. The macOS build helper validates a supplied `VCPKG_ROOT`, records it in build provenance, checks that the compiled executable contains an `arm64` slice, and executes `TextureMesh --help` to catch a binary that cannot load its linked libraries before writing trusted provenance.
4. Documentation now calls out the dependency check before building. No dependency installation, existing OpenMVS replacement, automatic cleanup, or original session overwrite has been introduced.

### Build-readiness finding

The user's terminal reports CMake 4.5.0, Git 2.39.3, Apple Clang 15.0.0 and Xcode Command Line Tools. The pinned upstream CMake requires CMake 3.18+ and automatically loads vcpkg only when `VCPKG_ROOT` is set. The user's value is currently unset. This is not by itself fatal if the required Boost, Eigen 3.4+, OpenCV, CGAL and VCG dependencies are already discoverable in system prefixes, but it means we must inspect the existing dependency setup before invoking CMake. The user's 31 GiB free disk space is recorded; no claim is made that this alone guarantees a dependency build will fit.

### CI and remaining uncertainty

- GitHub Actions on the final code-and-documentation revision completed successfully: **123 passed in 1.32 seconds**; integrated Python compilation and the build helper's Bash syntax check passed.
- CI verifies the Python contracts and shell syntax, not a native macOS OpenMVS compile.
- No fixed binary has yet been built on the user's Mac, and no texture-only run or visual colour/geometry acceptance has taken place. The reddish-brown fountain colour remains an objective, not a claimed result.

**Decision:** The isolated workflow is ready for dependency discovery, not yet ready for its native build. First locate an existing vcpkg toolchain or verify that the required libraries are already installed; only then run the separate build helper. Keep session-049 geometry, original texture package and runtime baseline frozen.

*Code review synthesis — ChatGPT*


## Entry 011 — ChatGPT / Native Build Without Local Dependency Changes — 2026-11-18

**New constraint from Saketh's Mac diagnostics:** Homebrew has Boost 1.92.0, CGAL 6.2.1 and Eigen 5.0.1, but OpenCV and VCG were not discoverable in the searched Homebrew/Conda locations; `VCPKG_ROOT` and a local `TextureMesh` executable are absent. The active checkout remains `perf/high-native-mesh-vectorization` at `b325bb55`, with six untracked logs/datasets/audit items. Do not switch that checkout, clean it, or install missing dependencies into the user's Mac as a first response.

### Artifact route investigated

- The upstream OpenMVS macOS ARM64 artifact from 2026-11-06 was successfully downloaded and inspected. It was built from develop commit `14ad2571aac650bfabf54296d31aabed7a19fdde`, which is 22 commits after the exact texture fix.
- That later history contains multiple intervening texturing changes (2x face-view computation, spatial patch grouping, custom atlas packing and later refine/texture changes). Therefore the newer prebuilt artifact is **not** a controlled replacement for the original session-049 TextureMesh path and was rejected for the first A/B test.
- Instead, PR #6 now adds a dedicated macOS arm64 GitHub Actions job that checks out the exact upstream fix `eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`, uses pinned vcpkg `9eec002df1cd109f8f104cd966a763daf9751489`, builds only `TextureMesh`, uses static OpenMVS/vcpkg linkage, verifies arm64 and Apple-system-only dynamic dependencies, runs `TextureMesh --help` with `DYLD_LIBRARY_PATH` unset, and uploads the executable plus provenance metadata as `TextureMesh-fixed-macos-arm64`.
- CI run `44210783972` has passed the regular 123-test reconstruction QA job. At the time of this entry, the native Apple-Silicon job has fetched vcpkg and the exact source commit and is still in CMake dependency configuration. No binary has yet been accepted or handed to the user.
- If the build succeeds, the executable's sidecar metadata will include the exact source commit and SHA-256 expected by `retexture_openmvs_session.py`; the user can download the artifact without installing OpenCV, VCG or vcpkg locally. The original Mac repository/workspace and all session-049 geometry/outputs remain untouched.



### Compatibility correction — macOS minimum version

Inspection of the upstream November 6 Apple-Silicon artifact showed its Mach-O executable was built with **minimum macOS 26.0**. It is therefore not a safe binary for a Sonoma (macOS 14) host, even though it is arm64. That artifact remains rejected.

The exact-fix CI workflow was updated to set both `MACOSX_DEPLOYMENT_TARGET=14.0` and `CMAKE_OSX_DEPLOYMENT_TARGET=14.0`. Before artifact upload it runs `xcrun vtool -show-build` and refuses the binary unless its Mach-O minimum OS reports macOS 14.0. The build metadata records `minimum_macos_version: 14.0`. New CI run: `44212503368`; dependency configuration is in progress. The earlier no-deployment-target run `44210783972` is superseded and should not be used.

**Decision:** Prefer the exact-source CI artifact over a large local vcpkg/Homebrew dependency install. Accept it only after the native build, arm64 check, runtime-launch check, artifact hash and sidecar provenance all pass. Then work from a separate Git worktree so the user's current checkout and untracked files remain intact. The actual six-photo texture A/B is still pending.

*Build-path review — ChatGPT*
