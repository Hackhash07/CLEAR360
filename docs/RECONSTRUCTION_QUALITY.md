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
edges, inconsistent face winding, disconnected component count, and the face
share held by components smaller than 0.1% of the complete mesh.

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
track support. Disconnected camera groups are warnings for review; they do not
automatically invalidate or delete a reconstruction.

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
