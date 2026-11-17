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
