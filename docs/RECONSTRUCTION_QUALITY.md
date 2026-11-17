# CLEAR360 Reconstruction Quality Report

CLEAR360 writes a backend-independent quality report after mesh export. The
same schema is used by the native Python reconstruction path and the COLMAP +
OpenMVS path.

## Output files

Each completed session may contain:

- `report.json`: the session report, including the `mesh_quality` object.
- `reconstruction_quality.json`: the detailed geometry and export audit.

## Status fields

- `geometry_status`: geometry-only heuristic: `healthy`, `review`, `poor`,
  or `unavailable`.
- `status`: overall status. Missing OBJ material references, an unreadable GLB,
  or low measured texture coverage can change an otherwise healthy geometry
  result to `review`.
- `score`: a 0–100 geometry heuristic, not a learned confidence or a guarantee
  of metric accuracy.
- `warnings`: specific conditions that should be inspected.

The geometry score considers degenerate faces, sliver triangles, non-manifold
edges, inconsistent face winding, disconnected component count, the face
share held by components smaller than 0.1% of the complete mesh, the fraction
of faces outside the largest connected component, and excessive boundary-edge
density on larger meshes. Boundary-edge density is a fragmentation signal,
not a watertightness test; small meshes are not penalized for being open.

The geometry-only thresholds are:

- `healthy`: score at least 90.
- `review`: score from 70 up to (but not including) 90.
- `poor`: score below 70.

The overall `status` can be `review` even when `geometry_status` is healthy,
for example when an OBJ references a missing texture or a GLB cannot be loaded.

## Important interpretation rules

1. **Watertightness is not mandatory.** Outdoor scenes, rooms, walls, terrain,
   furniture and partial object captures commonly produce valid open surfaces.
2. **Disconnected components are reported, not deleted.** Multi-part scenes can
   legitimately contain separate objects. Inspect small floating fragments
   before applying any cleanup.
3. **Texture export integrity is separate from geometric quality.** OBJ
   material-library and `map_Kd` references are resolved relative to their MTL
   files. GLB is opened as a scene to verify that its geometry can be read.
4. **Coordinate units may be arbitrary.** Unless the user supplies a reference
   distance, edge lengths and surface area are reported in coordinate units,
   not metres.
5. **An export succeeding is not proof that the reconstruction is complete.**
   Review the diagnostics and the model visually before using it downstream.

The quality audit is fail-soft: if a mesh cannot be audited, the report records
`unavailable` and the reconstruction output is preserved for investigation.
It does not silently modify or discard the generated geometry.

## Camera-overlap diagnostics

The OpenMVS registration summary also builds a camera co-visibility graph from
COLMAP `points3D.txt` tracks. A pair of registered cameras is connected when
they share at least 15 reconstructed 3D tracks, matching the threshold used by
the repository's M7.3.7 pose-graph audit. The report includes connected
components, isolated cameras, supported/possible pair counts and per-camera
track support. Before OpenMVS dense reconstruction, CLEAR360 now requires at
least 60% of uploaded images (and never fewer than three) to register, plus one
connected camera co-visibility graph. If this gate fails, dense MVS stops with
the failed image names and disconnected camera groups instead of creating a
misleading fragmented mesh. The registration diagnostics and sparse model are
preserved for investigation.

## Photo-set preflight

Before reconstruction, CLEAR360 audits the entire uploaded set:

- Verifies files can be decoded and have image metadata and quality assessments.
- Detects byte-identical uploads with SHA-256 hashes; duplicates are reported,
  never silently deleted.
- Aggregates blur, exposure/quality status and feature-density warnings.
- Compares normalized focal estimates so a resolution change alone does not
  look like a lens change; it also warns when camera models differ.
- Reports image dimensions and unusual aspect-ratio spreads.

Preflight states:

- `ready`: at least two distinct readable photographs and no preflight warnings.
- `review`: usable input exists, but warnings should be reviewed. The desktop
  workflow lets the user continue deliberately.
- `blocked`: fewer than two distinct, readable photographs remain.

Photo-set preflight does **not** claim to measure visual overlap. The report says
`overlap.status = not_measured` until COLMAP/SfM has produced camera poses and
shared 3D tracks. Post-SfM camera co-visibility diagnostics then report connected
camera groups, isolated views and shared-track support.

## Camera calibration selection

The OpenMVS backend selects COLMAP camera sharing from the upload audit:

- Consistent camera/lens metadata and matching image dimensions use one shared
  camera model.
- Mixed lens/device metadata or different image dimensions use an independent
  camera model per image.
- Original-upload EXIF camera metadata is retained when an upload is converted
  to JPEG and the conversion does not preserve EXIF.

COLMAP recommends sharing intrinsics when images come from the same physical
camera with identical settings. Different resolutions cannot safely be forced
into a single shared camera calibration; per-image calibration is used instead.
The selected strategy and reason are recorded under
`registration.camera_calibration` in the OpenMVS session report.

## Original uploads and SfM working images

CLEAR360 preserves the exact bytes uploaded by the user in the session's
`source_uploads/` directory. The `original_images/` directory is the working
set consumed by SfM and dense reconstruction:

- JPG, JPEG and PNG uploads are copied byte-for-byte.
- HEIC/HEIF, TIFF and BMP uploads are converted to JPEG for the current SfM
  readers. Original files remain archived.
- EXIF is copied to the normalized JPEG when the source decoder exposes it.
- `input_manifest.json` records the original filename, archive path, working
  filename, working format and whether conversion occurred.

Session reports use original filenames for user-facing diagnostics. Internal
camera-pose and working-image paths continue to use the normalized filenames
so every downstream stage references the same pixels.

## COLMAP and OpenMVS quality profile

The OpenMVS path uses the installed COLMAP CPU executable. The current feature
profile uses SIFT with affine-shape estimation, domain size pooling (DSP-SIFT),
and geometry-guided matching. These are COLMAP's documented options for
increasing robust feature matches; the command-line option namespaces are
version-specific and are covered by tests.

For the default Standard/High/Maximum profile, the OpenMVS depth-map pass
enables CPU confidence recalibration (postprocess-dmaps = 10, combining
short-gap interpolation with the CPU confidence pass). This adds a full-resolution
CPU pass but rejects weak depth estimates before fusion. Draft/Quick keeps the
faster interpolation-only setting (2).

The fusion pass requires at least three supporting views in Standard/High/
Maximum to reduce points supported by only two images, a known completeness
versus noise trade-off in OpenMVS. Draft/Quick retains two-view fusion for
coverage-oriented previews. Quality presets then run OpenMVS's camera-visibility
point-cloud filter (threshold -20) to remove points contradicted by registered
camera rays. The original unfiltered MVS/PLY files are retained, and the mesh
stage falls back to the unfiltered cloud if this optional filter fails. These
filters can remove weakly observed regions; they cannot infer surfaces that
were never photographed. The exact COLMAP and OpenMVS settings, plus whether
the visibility filter was actually applied, are recorded in
pipeline_configuration, visibility_filter and openmvs_pipeline.log.

The desktop panel distinguishes the OpenMVS executable minimum of six
photographs from a recommended 20–40+ well-overlapped capture sequence for
detailed object coverage. Passing image sharpness/uniqueness checks does not
prove that the images contain adequate overlap or sufficient viewpoints.

## Fast Preview mode and timing

The desktop phone-photo panel offers a separate `Fast Preview` preset and selects
it by default. It is a low-resolution, dense-point-cloud diagnostic pass intended
to target a sub-15-minute run on modest photo sets and hardware; that target is
not guaranteed and must be benchmarked on each device and input set. The current
OpenMVS preview profile uses an 800-pixel COLMAP feature/undistortion limit,
1,500 SIFT features, 640-pixel maximum MVS resolution at resolution level 1,
three source views, one PatchMatch iteration, one geometric-consistency iteration,
interpolation-only depth-map post-processing, and two CPU threads for COLMAP and
OpenMVS feature/mapping/dense processing.

Fast Preview retains the sparse registration and camera-connectivity gates and
uses two-view fusion. It skips CPU depth-confidence recalibration, visibility
filtering, triangular mesh reconstruction, mesh QA, full-resolution refinement,
camera-up alignment and both TextureMesh exports. It saves and opens the dense
PLY point cloud directly; there is no triangulated mesh or textured OBJ/GLB output.
The 3D viewer displays the point cloud as a point visual. The session report is
explicitly marked `PREVIEW_ONLY`, not `OPTIMAL` or final-quality. Use Standard or
High for the triangulated, refined and textured deliverable after reviewing the
preview and camera coverage.

Every external command now records its elapsed wall time in
`openmvs_pipeline.log` so the Fast Preview target can be evaluated stage by
stage. A preview that exceeds 15 minutes is a benchmark failure to investigate,
not a reason to silently lower quality or report a false time estimate.
