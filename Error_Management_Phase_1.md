# Error Analysis Plan: Reconstruction vs. Original (Phase 1)

## 1. Current State

The tooling as of 2026-03-04 provides:

**Reconstruction pipeline (`cut_list_reconstruction.py`):**
- `reconstruct_and_visualize()` parses a cut list `.txt` file, builds each wafer in its local frame (`Z = cylinder axis, M = X = (1,0,0)`), and assembles by the mark-alignment rule from `Wafer Reconstruction.txt`. Returns `{seg_name: wafer_list}`.
- `align_reconstruction_to_wafer(doc, seg_name, wafer_index_1based, rec_wafers)` applies a 6DOF (or 5DOF) rigid placement to the reconstruction Part so that a specified wafer's entry ellipse coincides with the original's. When LCS objects are present, the alignment is exact including M-axis spin. Without LCS, only the face normal (5DOF) is constrained.
- `report_exit_ellipse_discrepancy(doc, seg_name, wafer_index_1based, rec_wafers)` computes, for the same wafer after alignment: centroid distance (total/axial/lateral), normal angle, major-axis spin, and Blade° comparison. **Note**: `Nrm°` uses the LCS normal (reliable); `Bld°Δ` now uses the same LCS normal to select the correct exit face for `orig_blade_deg` extraction (face selection bug fixed 2026-03-04 — previously always returned the smaller-area face, giving θ_entry instead of θ_exit when θ_exit > θ_entry).

**Results table (`task_panel.py`, `_results_table`):**
- Columns: `Wfr | Seg | Aln | Ctrd mm | Axl mm | Lat mm | Nrm° | Spin° | Bld°Δ`
- Rows accumulate across multiple align+report calls without clearing, enabling multi-wafer and multi-alignment sweeps.
- The `Aln` column records whether 6DOF (LCS) or 5DOF (face fallback) was used.
- Color coding: amber if `Ctrd mm > 1.0` or `|Spin°| > 5°`; red if `Ctrd mm > 5.0` or `|Spin°| > 20°`.

**Original builder (`wafer_loft.py`, `driver.py`, `loft_segment.py`):**
- Convention B chord-bisector cutting planes at interior junctions; spine-tangent planes at endpoints.
- LCS objects `LCS_{seg}_{i}_1` (entry) and `LCS_{seg}_{i}_2` (exit) per wafer: `Z = chord direction` (inward), `X = major axis of the ellipse` (flipped so `normal.dot(chord) > 0`).
- Blade° written to cut list = `lift_angle_deg / 2.0` (symmetric average), plus separate `EntryBlade°` and `ExitBlade°` per face from `acos(|plane_normal · chord_dir_unit|)`. Length written as fractional inches plus exact `mm` column.
- Rot° computed exactly from `_lcs_signed_angle(major_axis_entry, major_axis_exit, chord_direction)` using the LCS major-axis vectors (LCS-based exact computation, replaces chord-polygon torsion approximation — candidate J fixed 2026-03-03).

---

## 2. Candidate Sources of Variation

Each candidate includes: (a) description, (b) expected signature in the results table, (c) how to test.

---

### A. Rounding in cut list values

**(a) Description.**
The builder writes Length in fractional inches rounded to nearest 1/16" (`_format_fractional_inches` in `driver.py`). Rounding error bounded by ±(1/32)" = ±0.794 mm. Blade° and Rot° written to 3 decimal places (±0.0005° error each). These values are the sole input to reconstruction; the original uses exact floating-point values internally.

**(b) Expected signature.**
- `Axl mm`: Non-zero on every wafer. Bounded by ±0.794 mm. Independent per wafer (random sign, not cumulative).
- `Nrm°`, `Spin°`: Negligible (angular rounding is ±0.0005°).
- Pattern: `Axl mm` varies randomly per wafer within ±0.794 mm, not growing with index.

**(c) How to test.**
From the compare_reconstruction log, read `Len CL` and `Len Or` for each wafer. Compute `Δlen = (Len Or − Len CL) × 25.4` mm. Compare to `Axl mm` in the results table for the same wafer. If `Axl mm ≈ Δlen`, rounding explains the axial error.

---

### B. Mark direction sign convention mismatch (LCS X-axis vs. spec +M)

**(a) Description.**
The spec defines +M as the endpoint of the major axis that protrudes toward the entry side on the entry face (lower Z in the wafer local frame). The LCS X-axis is set from OCC's `edge.Curve.XAxis`, whose sign is not guaranteed to equal the spec's +M. After extraction, `generate_wafers()` (lines 997–999 of `wafer_loft.py`) flips `major_axis` when the face normal opposes the chord direction to maintain right-hand rule, but this does not guarantee agreement with the spec's +M convention.

If the sign is wrong by 180°, every wafer's mark-alignment placement is rotated 180° about the face normal from where it should be. The reconstruction is self-consistent but systematically inverted relative to the original.

**(b) Expected signature.**
- `Spin°`: Constant ~±180° on every wafer (even when alignment is 6DOF).
- `Nrm°`: Near zero (face orientation unaffected by 180° spin about face normal).
- `Ctrd mm`: Near zero axially; small lateral offset on non-symmetric wafers.
- Pattern: Constant across all wafer indices — does not grow.

**(c) How to test.**
1. Use 6DOF LCS alignment. Align to wafer 1 and report wafer 1 exit. A constant ~±180° `Spin°` confirms this candidate.
2. Cross-check: in `_get_orig_entry_frame`, verify that `lcs.XAxis.dot(spec_mark_dir) > 0`. Compute the spec +M direction from the reconstruction geom dict `entry_mark_dir` after applying the reconstruction's placement.

---

### C. S-curve inflection encoding (torsion reversal)

**(a) Description.**
At an S-curve inflection point, the physical direction of blade tilt reverses. The spec encodes this reversal by adding ~180° to Rot° for that cut, keeping Blade° ≥ 0. The builder computes Rot° from the chord-polygon torsion. At a true planar inflection, this torsion is near 0°, not 180°. If the builder does not explicitly detect the torsion reversal and add 180°, the reconstruction assembles subsequent wafers with the wrong orientation.

**(b) Expected signature.**
- `Spin°`: Sudden large spike (~±180°) at the inflection wafer index k; near zero before k.
- `Ctrd mm`: Grows monotonically from wafer k onward (accumulates after the misoriented wafer).
- `Nrm°`: May remain near zero before k; jumps at k.
- Pattern: Step-function error. Clean results for wafers < k; rapidly growing errors for wafers > k.

**(c) How to test.**
1. Use a sinusoidal or trefoil curve with at least one inflection. Note the inflection wafer index from the original geometry.
2. Build the results table by aligning to wafer 1 and sweeping exits from wafer 1 to N. Identify where `Spin°` spikes.
3. Inspect cut list Rot° at that row: should be near ±180° if the reversal was correctly encoded.

---

### D. Chord-bisector (Convention B) vs. flat-cylinder reconstruction

**(a) Description.**
Convention B: interior cutting planes have normals equal to `normalize(d_prev + d_next)` (chord bisector). The reconstructor builds each wafer with the spec's per-cut Blade° applied symmetrically. On a non-circular path, the bisector angle at a junction is generally not equal to the blade angle computed from the wafer dihedral. The error is O(turning_angle) and roughly proportional to local curvature.

**(b) Expected signature.**
- `Nrm°`: Small non-zero, present on every interior wafer. Larger at high-curvature regions.
- `Axl mm`: Small secondary contribution (~`R · Δθ · sin(θ)`) where Δθ is the blade angle error.
- `Spin°`: Not directly affected by this candidate.
- Pattern: Smooth, correlated with local curvature. Zero on a circle; grows at sinusoidal peaks.

**(c) How to test.**
1. Build a pure circle: `Nrm°` should be near zero. Any non-zero value isolates other candidates.
2. Build a sinusoidal curve. Plot `Nrm°` vs. wafer index and compare to a curvature profile. Correlation confirms candidate D.

---

### E. Length definition: chord vs. axial center-to-center

**(a) Description.**
The spec defines Length as the center-to-center distance along the cylinder axis (axial distance L). The builder writes `chord_length` = Euclidean distance between the two cutting-plane center points = `|plane2['point'] − plane1['point']|`. For a wafer with nonzero blade angles, this is larger than the axial projection by a factor of `1/cos(θ_avg)`. If the builder writes chord distance and the reconstructor uses it as the axial length L, each wafer's exit center is shifted axially by `L · (1 − cos(θ_avg)) ≈ L · θ_avg² / 2`.

**(b) Expected signature.**
- `Axl mm`: Systematic positive offset, proportional to `θ²` (grows with Blade°). Distinguished from rounding (candidate A) by its dependence on Blade° rather than random variation.
- `Lat mm`: Near zero.
- `Nrm°`: Near zero.
- Pattern: Larger on high-blade wafers; near zero on flat-circle wafers.

**(c) How to test.**
For each wafer, compute `predicted_axl = L_cut_list × (1 − cos(theta_exit_deg_in_radians)) × 25.4` mm. If this matches `Axl mm`, candidate E is confirmed. Alternatively, re-run reconstruction with `L_axis = L_chord × cos(theta_avg)` and check if `Axl mm` drops to near zero.

---

### F. Initial blade angle for closed curves

**(a) Description.**
For a closed curve, wafer 0's entry face is defined by the last cut row, not an implicit flat circle. The reconstruction must know `theta_entry_0` (the blade angle of that entry face) to initialize the joint state. The builder writes `Initial blade: {value:.3f}°` to the cut list header when `closed = True`. If missing, the reconstruction falls back to `all_rows[-1]['blade_deg']` (last row's blade angle) with a warning logged.

If `theta_entry_0` is wrong, every wafer's placement is based on an incorrect initial frame, causing errors that appear immediately on wafer 1 and grow rapidly.

**(b) Expected signature.**
- Large errors beginning on wafer 1 (unlike most candidates, which appear gradually).
- `Nrm°` and `Spin°` large on wafer 1, growing.
- Log output: "No 'Initial blade' in cut list; using last row blade" if the fallback is triggered.

**(c) How to test.**
1. Check the log for the fallback warning.
2. Compare `Initial blade:` in the cut list header to `theta_entry_0` logged during reconstruction.
3. Test sensitivity: modify `initial_blade_deg` by ±1° and rerun; observe how quickly the table values diverge.

---

### G. LCS derivation accuracy and sign

**(a) Description.**
The LCS Z-axis is set to the chord direction (after flipping if needed). The LCS X-axis is set to `major_axis` extracted from OCC's `edge.Curve.XAxis`, sign-flipped when the normal opposed the chord direction (to maintain right-hand rule). This flip ensures Z-consistency but does not guarantee the spec's +M sign convention.

Separately: `_build_rotation_from_frame` constructs a rotation from two orthonormal pairs. If the pairs are not exactly orthogonal (floating-point), the rotation is slightly non-unitary. For typical inputs this error is < 1e-10° and negligible.

**(b) Expected signature.**
- `Nrm°` non-zero even for the aligned wafer's own exit: indicates LCS Z-axis inaccuracy.
- `Spin°` ≈ ±180° consistently: LCS X-axis sign is inverted (overlaps with candidate B).
- `Aln` column: "5DOF" means LCS was absent or unusable; spin is then unconstrained.

**(c) How to test.**
1. Check `Aln` column: "6DOF (LCS)" confirms LCS was used for full alignment.
2. In FreeCAD console: `lcs = App.ActiveDocument.getObject("LCS_seg_0_1"); print(lcs.Placement.Rotation.multVec(App.Vector(0,0,1)))` — should match chord direction.
3. Verify `lcs.XAxis.dot(chord_dir) ≈ 0` (X perpendicular to chord).

---

### H. Numerical floating-point accumulation

**(a) Description.**
Each wafer's placement is derived from the previous wafer's exit frame. Each step involves ~15 floating-point operations for matrix multiplication plus normalization. Over N wafers, accumulated double-precision error is O(N × 2 × 1e-15), bounded by ~3e-13 for 100 wafers. This is far below the results table's 4 decimal place resolution (0.0001 mm).

**(b) Expected signature.**
- `Ctrd mm` growing linearly with wafer index at sub-nanometer rates.
- In practice: zero visible effect.

**(c) How to test.**
Build a simple circle (all equal wafers). Check `Ctrd mm` growth rate vs. index. Any growth visible to 4dp (> 0.0001 mm) is NOT due to floating-point alone.

---

### I. Major axis sign ambiguity in face extraction (fallback, no LCS)

**(a) Description.**
When LCS objects are absent and alignment falls back to face detection, the major axis is extracted from OCC's `edge.Curve.XAxis`, which does not guarantee a consistent sign across different wafers or sessions. If the sign is random, `Spin°` shows ±180° on some wafers but not others, with no predictable pattern.

For circular faces (Blade° = 0), no major axis is extractable, so `major_world = None` and alignment falls back to 5DOF, leaving `Spin°` unconstrained.

**(b) Expected signature.**
- `Aln` = "5DOF" for zero-blade wafers.
- `Spin°` ≈ ±180° on some wafers and near 0° on others, with no pattern (random sign per OCC call).
- Distinguishable from candidate B (systematic ±180°) by randomness.

**(c) How to test.**
1. Check `Aln` column: "5DOF" means spin is unconstrained.
2. Rebuild the document from scratch and repeat the same alignment. If `Spin°` changes sign between sessions, OCC sign randomness is confirmed.
3. Compare results with and without LCS: LCS rows should be consistent; face-only rows may vary.

---

### J. Rot° derivation accuracy (chord-polygon torsion approximation)

**(a) Description.**
The builder computes `rotation_angle_deg` (Rot°) from three consecutive chord vectors using a cross-product dihedral formula. This approximates the torsion of the chord polygon, not the torsion of the smooth curve. The approximation error is O(h²/R_curve) where h is the chord length. For chord ~6 mm and curvature radius ~50 mm, the error per wafer is ~0.04°.

This is separate from candidate C (inflection point sign reversal): candidate J produces small gradual errors at high curvature; candidate C produces a sudden large error at an inflection.

**(b) Expected signature.**
- `Spin°`: Small non-zero, grows with local curvature. Distinct from candidate C by being gradual, not sudden.
- Pattern: Proportional to `(chord / R_curve)²`. Smaller with shorter chords (more wafers).

**(c) How to test.**
1. Build a helix (constant non-zero torsion). Compare `Spin°` per wafer to analytically expected torsion. The residual after subtracting exact torsion is the approximation error.
2. Reduce `max_chord` and rerun. If `Spin°` drops as chord², this candidate is confirmed.

---

## 3. Diagnostic Procedure

Work from simplest geometry to most complex. The results table accumulates rows across runs.

**Step 1: Baseline on a simple circle (all candidates should be near-zero).**

Build a planar circle. Reconstruct. Align to wafer 1. Report exits for wafers 1, 2, 5, N/2, N.

Expected: `Ctrd mm < 0.1`, `Nrm° < 0.1°`, `Spin°` consistent if LCS present.

- Non-zero constant `Axl mm` on every wafer → candidate A (rounding).
- `Spin° ≈ ±180°` on every wafer → candidate B or I (check `Aln` column).
- `Ctrd mm` growing with index → investigate G or J.
- Near-zero everywhere → clean baseline; proceed to Step 2.

**Step 2: Sweep multiple alignment wafers on the circle.**

Align to wafer 1, then N/4, then N/2 in separate calls. Compare the three result rows.

- `Ctrd mm` proportional to wafer index difference → systematic cumulative candidate (A, E, J).
- `Nrm°` same for all three alignments → intrinsic per-wafer error, not cumulative.
- `Spin°` consistently ±180° → candidate B confirmed.

**Step 3: Isolate rounding vs. length definition (candidates A and E).**

From the compare_reconstruction log, compute `Δlen` for each wafer. Compare to `Axl mm`:
- `Axl mm ≈ Δlen` → candidate A (rounding) explains the error.
- `Axl mm > Δlen` and correlated with Blade° → candidate E (chord vs. axial length).
- Compute `predicted_axl_E = L_chord × (1 − cos(theta_exit))` and compare.

**Step 4: Chord-bisector effect (candidate D) on a sinusoidal curve.**

Build a sinusoidal curve. Align to wafer 1. Report exits at curvature peaks and inflections.

- `Nrm°` larger at peaks, smaller at inflections → candidate D confirmed.

**Step 5: Inflection encoding (candidate C) on trefoil.**

Build trefoil. Align to wafer 1. Sweep exits from wafer 1 to N. Look for a sudden large jump in `Spin°` at wafer k. Inspect cut list Rot° at row k — should be near ±180°. If Rot° is small there, the reversal was not encoded → candidate C confirmed.

**Step 6: Closed-curve initialization (candidate F).**

Check cut list header for "Initial blade:" line. Compare to `theta_entry_0` in the reconstruction log. Missing line → check for fallback warning. Modify `initial_blade_deg` by ±1° and rerun to measure sensitivity.

**Step 7: LCS quality (candidate G).**

For rows with `Aln = "6DOF (LCS)"`: any exit `Spin° ≠ 0` after exact 6DOF alignment reveals true parameter errors, not alignment artifacts. Verify LCS Z/X axes as described in candidate G.

For rows with `Aln = "5DOF"`: `Spin°` is unconstrained — investigate candidate I or rebuild with LCS enabled.

---

## 4. Correction Strategies

### A. Rounding in cut list values

Change `_format_fractional_inches` in `driver.py` to output decimal inches with 4+ decimal places, or add a supplementary exact-value column. Update `parse_cut_list` in `cut_list_reconstruction.py` to accept decimal values.

**Files**: `driver.py` `_write_lofted_segment_block`; `cut_list_reconstruction.py` `_parse_fractional_inches`.

---

### B. Mark direction sign convention mismatch

In `generate_wafers()` (`wafer_loft.py`), after extracting `major_axis1/2`, apply a sign check to ensure the LCS X-axis points toward the spec's +M (toward the lower-Z endpoint on the entry face, toward the higher-Z endpoint on the exit face). Compute `mark_candidate = center + R × major_axis_dir`; compare its Z-component against the plane's Z to determine whether the sign needs flipping.

**File**: `wafer_loft.py` `generate_wafers()`, lines 990–1028.

---

### C. S-curve inflection encoding

In `generate_all_wafers_by_slicing()` (`wafer_loft.py`), detect torsion sign reversals: when the cross-product plane normal flips direction between consecutive chord triplets (`plane_A_normal.dot(plane_B_normal) < 0`), add 180° to the computed `rotation_angle_deg`. This encodes the M-direction reversal in Rot° rather than leaving it as an implicit sign change.

**File**: `wafer_loft.py` rotation angle calculation loop (approximately lines 822–839).

---

### D. Convention B vs. flat-cylinder reconstruction

Options (ascending complexity):
1. **Accept the approximation**: characterize its magnitude; for typical geometry it is ~0.06° per wafer at medium curvature.
2. **Add ExitBlade° column** to the cut list and use it in reconstruction. Requires changes to both builder and parser.
3. **Use midpoint blade angles**: compute entry and exit blade angles for each wafer separately and use their average in reconstruction.

**Files**: `driver.py` `_write_lofted_segment_block`; `cut_list_reconstruction.py` `parse_cut_list` and `reconstruct_segment`.

---

### E. Length definition: chord vs. axial

In `_write_lofted_segment_block`, write the axial projection `L_axis = (plane2['point'] − plane1['point']).dot(chord_direction)` instead of the Euclidean chord distance. This ensures the reconstructor's L matches the spec's center-to-center axial definition.

**File**: `driver.py` `_write_lofted_segment_block`, Length computation.

---

### F. Initial blade angle for closed curves

Verify that `initial_blade_deg` written to the cut list header matches the spec's definition. Ensure the value is written with sufficient precision (increase from 3dp to 5dp for tight curves). Add explicit validation in `parse_cut_list` to require the `Curve closed:` line rather than inferring from curve type name.

**Files**: `driver.py` (precision); `cut_list_reconstruction.py` `parse_cut_list`.

---

### G. LCS derivation accuracy

If LCS X-axis sign is wrong: apply the correction from B (mark-direction sign check). If further inaccuracy is found in Z/X orthogonality: add an explicit Gram-Schmidt step after LCS construction (`x = x − z*(x·z); x = normalize(x)`).

**File**: `wafer_loft.py` `generate_wafers()`.

---

### H. Numerical floating-point accumulation

No correction needed. Negligible at double precision for all practical wafer counts (< 10,000 wafers).

---

### I. Major axis sign ambiguity (face extraction fallback)

In `_get_orig_entry_frame` and `_get_orig_exit_frame`, after extracting `x_local` from OCC, apply the same mark-direction sign check as correction B. For circular faces (Blade° = 0), continue to return `major_world = None` (5DOF alignment is correct; report `Spin°` as "-" rather than a spurious value).

**File**: `cut_list_reconstruction.py` `_get_orig_entry_frame()` and `_get_orig_exit_frame()`.

---

### J. Rot° derivation accuracy

Replace the chord-polygon torsion approximation with a direct LCS-based computation: `rotation_angle_deg = signed_angle(lcs_{i}_1.XAxis, lcs_{i}_2.XAxis, chord_direction)`. The LCS objects are already built and stored at this point in `generate_wafers()`; the major axis vectors from the face ellipses are available in `geometry['ellipse1/2']['major_axis_vector']`.

**File**: `wafer_loft.py` `generate_all_wafers_by_slicing()`, rotation angle computation loop.

---

## 5. Priority Ranking

Based on expected impact for a typical trefoil:

| Rank | Candidate | Likely Impact | Effort | Status |
|------|-----------|---------------|--------|--------|
| 1 | B — Mark direction sign | ±180° spin on all wafers | Low | REQUIRES TESTING |
| 2 | C — Inflection encoding | Large step error at S-curves | Medium | NOT A BUG (see §7) |
| 3 | J — Rot° approximation | Small growing spin error | Medium | CONFIRMED — FIXED (LCS exact) |
| 4 | A — Length rounding | ±0.8 mm axial per wafer | Low | CONFIRMED — FIXED (mm col) |
| 5 | E — Chord vs. axial length | Systematic axial offset | Low | NOT A BUG (see §7) |
| 6 | D — Convention B mismatch | Small normal error at curves | High | CONFIRMED — FIXED (EntryBlade°/ExitBlade°) |
| 7 | F — Closed-curve initialization | Large error if triggered | Low | REQUIRES TESTING |
| 8 | I — OCC sign randomness | Random ±180° (fallback only) | Low | REQUIRES TESTING |
| 9 | G — LCS numerical accuracy | Sub-0.01° (negligible) | Low | REQUIRES TESTING |
| 10 | H — Floating-point accumulation | Sub-nanometer (negligible) | None | NOT A BUG (see §7) |

---

## 7. Confirmed Code Behaviors (2026-03-03 Audit)

Code inspection (as of this date) establishes the following status for each candidate.

---

### A — CONFIRMED: Length rounds to 1/16"

`src/driver.py` `_format_fractional_inches()`: `sixteenths = round(frac * 16)` → maximum rounding error ±(1/32)" = ±0.794 mm per wafer.

**Fix implemented:** A `mm` column is now written alongside the fractional column (e.g., `35.05`). The parser reads the `mm` column when present and uses it (converted to inches) as `row['length']`, falling back to fractional parse when absent. No change to the fractional format (preserves human readability).

---

### D-partial — CONFIRMED: Single symmetric Blade° per wafer

`src/driver.py` `_write_lofted_segment_block()`: `blade_angle = lift_angle_deg / 2.0`.

`lift_angle_deg` is the dihedral angle between the two cutting-plane normals. Dividing by 2 assumes entry tilt = exit tilt, which holds exactly only for a symmetric wafer on a uniform curve. For a curved path with Convention B bisectors, the plane normal at a junction makes a different angle with each adjacent chord direction: `theta_entry ≠ theta_exit` in general.

**Fix implemented:**
- `src/wafer_loft.py` `generate_all_wafers_by_slicing()`: after each geometry dict is built, computes `theta_entry_deg` and `theta_exit_deg` from `acos(|plane_normal · chord_dir_unit|)` and stores them in the geometry dict.
- `src/driver.py` `_write_lofted_segment_block()`: writes `EntryBlade°` and `ExitBlade°` columns alongside the existing `Blade°` (retained for backward compat).
- `src/cut_list_reconstruction.py` `parse_cut_list()`: reads `EntryBlade°`/`ExitBlade°` columns if present and stores them as `row['theta_entry_deg']`/`row['theta_exit_deg']`. `reconstruct_segment()` uses them directly when available, instead of the symmetric approximation.

---

### C — NOT A BUG: Planar S-curve inflection gives Rot° = 0°

For a planar forward-traveling S-curve, consecutive chord triplets at the inflection point remain coplanar. Both cross-product plane normals point in the same direction, so `rotation_angle_deg ≈ 0°`. This is **geometrically correct**: the M-axis does not need to flip for a smooth planar S-curve; curvature reversal without a chord hairpin does not require a 180° encoding. The 180° branch is only triggered when `acos(plane_A · plane_B) ≈ 180°`, which occurs for a true hairpin (chord physically reverses). Candidate C is dismissed as a code bug for typical planar sinusoidal use cases.

---

### E — NOT A BUG: chord_length IS the axial L

`chord_length = |plane2['point'] − plane1['point']|` is the Euclidean distance between the two cutting-plane center points. In the reconstruction's local coordinate frame, the cylinder Z-axis IS the chord direction (axis from entry center to exit center). Therefore `L = chord_length` exactly. No axial projection correction is needed.

---

### H — NOT A BUG: Floating-point accumulation negligible

At double precision, accumulated error over 100 wafers is bounded by ~3 × 10⁻¹³ mm — far below any measurable threshold. No action required.

---

### J — CONFIRMED AND FIXED: Rot° chord-polygon torsion approximation (2026-03-03)

The chord-polygon torsion approximation was replaced with an exact LCS-based computation in `generate_all_wafers_by_slicing()`. The rotation angle is now computed as `_lcs_signed_angle(major_axis_entry, major_axis_exit, chord_direction)` using the major-axis vectors stored in the geometry dict. The same normal-flip applied in `generate_wafers()` (`if normal.dot(chord_unit) < 0: major_axis = -major_axis`) is applied before the call.

---

### Bld°Δ face selection bug — FIXED (2026-03-04)

`orig_blade_deg` in `report_exit_ellipse_discrepancy` used `_find_planar_faces` (area-sorted, largest first) plus a chord dot-product test to select the exit face. The dot product is always negative regardless of ordering (face outward normals always oppose the chord to the other face), so the code always picked `fb` — the smaller-area face. When θ_exit > θ_entry the exit face is larger (= `fa`), so `fb` = entry face was incorrectly returned. Symptom: `Nrm° = 0.00` (LCS-based, correct) but `Bld°Δ = θ_entry − θ_exit < 0`.

**Fix**: replaced the area-sort/dot-product selection with `exit_face = max(planar, key=lambda f: abs(_face_normal(f).dot(orig_exit_inward)))`, using the already-correct LCS-based `orig_exit_inward` vector as the reference.

---

### B, F, G, I — REQUIRES RUNTIME TESTING

These candidates cannot be confirmed or dismissed from code inspection alone. They require the alignment + report tool to produce data on actual builds. See Section 3 for the diagnostic procedure.

---

## 8. Physical Build Variance

### Statistical model

Physical woodworking introduces per-cut independent errors that accumulate as a random walk over N cuts. Assuming independent Gaussian errors with 1σ = one standard deviation:

| Error source | Per-cut 1σ | Cumulative at N cuts | Physical unit |
|-------------|-----------|---------------------|--------------|
| Blade tilt | σ_b | σ_b × √N | degrees normal-angle |
| Blade tilt (lateral) | R × sin(σ_b_rad) | R × sin(σ_b_rad) × √N | mm lateral shift |
| Rot spin | σ_r | σ_r × √N | degrees spin |

Measured tool precision (2026-03-04):
- Blade angle set with a digital angle ruler accurate to ±0.1°: uniform error ±0.1° → **σ_b = 0.1°/√3 ≈ 0.06°**
- Rotation set with a simple protractor accurate to ±2°: uniform error ±2° → **σ_r = 2°/√3 ≈ 1.15°**

Rotation error dominates blade error by ~20×; improving the protractor yields far greater benefit than improving blade angle setting.

For a 2" diameter (1" radius) cylinder and N = 20 wafers:
- Normal: ±0.06° × √20 ≈ **±0.27°**
- Spin: ±1.15° × √20 ≈ **±5.1°**
- Lateral: 25.4 mm × sin(0.06°) × √20 ≈ **±0.12 mm**

### Implementation

**`src/driver.py`** writes variance defaults in the cut list header:
```
  Blade° variance (1σ): ±0.500°
  Rot° variance (1σ): ±2.000°
```

**`src/cut_list_reconstruction.py`** `parse_cut_list()` reads these lines into `segment_data['sigma_blade_deg']` and `segment_data['sigma_rot_deg']`.

**`src/cut_list_reconstruction.py`** `report_exit_ellipse_discrepancy()` accepts `sigma_blade_deg` and `sigma_rot_deg` parameters (defaults 0.06° and 1.15°) and includes a `build_sigma` key in its return dict with cumulative 1σ bounds.

**`src/gui/task_panel.py`** has a "Build Variance" section with "Blade σ°" and "Rot σ°" spin boxes (defaults 0.06°, 1.15°). After each align+report call, the variance impact label updates:
```
Build σ (1σ, N=20):  Normal ±0.27°  Spin ±5.1°  Lateral ±0.12 mm
```

### Interpretation

The variance bounds should be read as: _if every reconstruction error is within the code's computed bounds, and physical build variance is within these 1σ values, the expected cumulative positional error at the end of the assembly is dominated by physical build variance, not code errors._ The results table columns (`Nrm°`, `Spin°`, `Ctrd mm`) are directly comparable to these bounds. Values within ±1σ are consistent with expected physical variation; values beyond ±3σ indicate genuine code errors.

---

## 9. Correction Point Analysis (added 2026-03-04)

### Concept

Spin error accumulates as a random walk: after k cuts, the expected 1σ error is σ_r × √k. A **correction point** resets the walk to zero. It is created by splitting one wafer at its midpoint with a perpendicular (blade = 0°) cut, creating a flat circular joint. Because a flat circle is rotationally symmetric, the cylinder can be freely re-indexed at that joint during the physical build to absorb any accumulated spin error before proceeding.

The split produces two modified rows in the cut list:

| Original | Half A | Half B |
|----------|--------|--------|
| θ_entry, θ_exit, Rot°, L | θ_entry, 0°, 0°, L/2 | 0°, θ_exit, Rot°+ε, L/2 |

where ε is the correction rotation measured at build time.

### Correction Analysis Tool

**`src/analyzer/correction_analysis.py`** — pure-numeric, no FreeCAD dependency.

`analyse_correction_points(rows, sigma_rot_deg, top_n=5)` scores every possible split wafer and every possible split pair. Sorting is by `max_err_deg` (primary) and `max_blade_deg` (secondary tiebreaker).

The "Correction Analysis" button in the task panel (next to "Build from Cut List") runs this analysis on the last loaded cut list using the current "Rot σ°" spinner value, and prints the report to the log pane.

### Reading the Correction Analysis Report

**Single correction table** — each row is a candidate wafer to split:

| Column | Meaning |
|--------|---------|
| `Rank` | 1 = best |
| `Cut` | Cut number from the cut list |
| `SplitAt` | Effective correction position in wafer units (cut − 1 + 0.5) |
| `Before` | Wafers in the section before the correction |
| `After` | Wafers in the section after the correction |
| `MaxErr` | σ_rot × √(longer section) — the 1σ spin error at the worst point |
| `MaxBlade` | Larger of entry/exit blade angle — higher = Z-axis mark easier to read |
| `entry / exit` | Individual blade angles of the two faces of the split wafer |

**Two-correction table** — each row is a candidate pair of wafers to split:

| Column | Meaning |
|--------|---------|
| `Cuts` | The two cut numbers (A, B) |
| `Sec1/2/3` | The three section lengths in wafers |
| `MaxErr` | σ_rot × max(√sec1, √sec2, √sec3) |
| `Blades A/B` | MaxBlade for each of the two split wafers |

**Ideal positions**: one correction at N/2 (±0° improvement); two corrections at N/3 and 2N/3. Deviations from ideal depend on which wafers have blade angles making them suitable candidates.

**Example** (N=20, σ_r=1.15°): no corrections → max error 5.14°; best single correction (wafer 10) → 3.24°; best double correction (wafers 6+13) → 2.65°.

### Measuring and Applying the Correction

**Z-axis mark scheme** (recommended): before each cut, scribe a mark at the highest point of the cylinder in the saw jig. This records the rotation state at every cut using gravity as a free reference. At the correction point's flat face:

1. Read the Cylinder° value for that cut from the cut list — this is the nominal mark angle from the build's reference vertical.
2. Place a digital angle gauge against the most recent elliptical face (last face before the flat cut) and measure the actual major-axis angle from vertical.
3. Correction ε = actual − nominal.
4. Before making the next cut after the flat joint, rotate the cylinder by −ε from its current position.
5. Update the second-half row's Rot° to Rot°_original + ε.

**Distance measurement** (supplementary): three fixed reference points in the lab give 3D position, which quantifies accumulated position drift (blade/length errors). This is complementary to spin correction, not a substitute.

### Relationship to the User Guide

This section describes _what tools are available_ and _how to interpret their output_. The physical build procedure — how to use the saw jig, how to scribe marks, how to set up reference points, and the step-by-step correction workflow — belongs in a separate User Guide document.

---

## 6. Critical Files

| File | Relevance |
|------|-----------|
| `src/cut_list_reconstruction.py` | Core reconstruction, alignment, and discrepancy reporting — all diagnostic procedures run through here |
| `src/wafer_loft.py` | Original builder: cutting planes (Convention B), LCS creation, Rot°/Blade° derivation — source of B, C, D, G, J |
| `src/driver.py` | Cut list writer: Length formatting, Blade°/Rot° output, initial blade — source of A, E, F |
| `src/gui/task_panel.py` | Results table and alignment UI: where diagnostic sweeps are run interactively |
| `Wafer Reconstruction.txt` | Authoritative spec: sign conventions, mark definitions, assembly rule |
| `src/analyzer/correction_analysis.py` | Correction point candidate selection — pure-numeric, no FreeCAD dependency |

---

End of file.
