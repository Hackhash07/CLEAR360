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
edges, inconsistent face winding, and disconnected component count.

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
