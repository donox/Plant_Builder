# Error Management --- Reconstruction Analysis Plan (Phase 1)

## 1. Purpose

PlantBuilder generates:

1.  An original FreeCAD structure composed of sequential wafer solids.
2.  A cut list derived from that structure.
3.  A reconstructed structure built solely from the cut list.

The reconstructed structure currently deviates from the original and
does not close properly in closed-loop cases (e.g., trefoil).

**Objective (Phase 1):**\
Identify where and how the reconstructed geometry diverges from the
original geometry by comparing corresponding wafers one at a time and
quantifying cumulative drift and closure error.

This phase focuses strictly on algorithmic / geometric correctness ---
not physical build tolerances.

------------------------------------------------------------------------

## 2. Geometry Model

### 2.1 Wafer Definition

Each wafer is a FreeCAD solid created by slicing a cylinder with two
oblique planes.

Each wafer consists of: - Two planar end faces (entry and exit) - One
cylindrical side surface

Wafers are not fused into a compound. They are placed independently so
that adjacent faces coincide.

------------------------------------------------------------------------

## 3. Reconstruction Model

Reconstruction follows the Wafer Reconstruction Specification:

-   Each cut list row defines a cut plane.
-   A cut defines:
    -   The exit face of wafer *i*
    -   The entry face of wafer *i+1*
-   Reconstruction uses:
    -   Cylinder radius (from cut list header)
    -   Blade°
    -   Rot°
    -   Length (center-to-center distance along cylinder axis)

Not used in reconstruction: - Cylinder°

Rules: - Blade° must be ≥ 0. - If negative blade is encountered →
error. - If header is missing cylinder diameter: - Assume default
diameter 2.25" (radius 1.125") - Flag as error (header missing).

------------------------------------------------------------------------

## 4. Scope

### Phase 1 (This Phase)

-   Compare original vs reconstructed wafers
-   Identify where mismatch begins
-   Quantify per-wafer error
-   Quantify cumulative drift
-   Quantify final closure gap

### Phase 2 (Later)

-   Monte Carlo simulation of physical build tolerances:
    -   Blade variance \~0.5°
    -   Rotation variance up to \~2°
    -   Independent/random per cut

Phase 2 is explicitly out of scope for now.

------------------------------------------------------------------------

## 5. User Workflow (FreeCAD)

Add a new action in the PlantBuilder Task Panel:

Button: **Analyze Reconstruction**

Preconditions: - Active document must contain exactly two top-level
Parts: - `<Name> Part` (original) - `<Name> Reconstructed`
(reconstructed)

If more than two Parts exist → error.

Default visualization mode: **overlay**.

------------------------------------------------------------------------

## 6. Analyzer Operation

### 6.1 Step Mode

Analyzer operates wafer-by-wafer:

For wafer index *i*: - Highlight original wafer *i* - Highlight
reconstructed wafer *i* - Compute comparison metrics - Output metrics to
console - Allow Next / Previous navigation

------------------------------------------------------------------------

## 7. Per-Wafer Feature Extraction

For each wafer solid (original and reconstructed):

Extract: - Entry face - Exit face - Face centers - Face normals

If ellipse geometry available: - Major-axis direction - Detect sign
ambiguity

If major-axis / marking ambiguity cannot be resolved → error.

------------------------------------------------------------------------

## 8. Per-Wafer Comparison Metrics

For wafer *i*, compute:

### Entry Face

-   Normal angular difference (degrees)
-   Center-to-center distance

### Exit Face

-   Normal angular difference (degrees)
-   Center-to-center distance

If solids do not align, compare faces independently.

------------------------------------------------------------------------

## 9. Cumulative Drift Metrics

Up through wafer *i*:

-   Exit face center drift:
    -   Vector difference
    -   Magnitude
-   Exit face normal drift:
    -   Angular difference

This identifies where divergence begins and how it accumulates.

------------------------------------------------------------------------

## 10. Closure Metric (Closed Curves)

For closed structures:

Compute distance between: - Reconstructed final exit face center -
Original wafer 1 entry face center

Phase 1 metric: - Center distance only

Future expansion: - Normal comparison - Axis comparison

------------------------------------------------------------------------

## 11. Error Conditions (Phase 1)

Flag as errors:

-   Negative Blade° in cut list
-   Missing cylinder diameter in header (after default applied)
-   More than two top-level Parts
-   Non-trailing degenerate wafers
-   Unresolved axis / marking ambiguity

Degenerate wafers at the end (due to over-requested count) may be
ignored.

------------------------------------------------------------------------

## 12. Outputs

### Required: Console Output

For each wafer:

-   Wafer index
-   Entry:
    -   Normal Δθ
    -   Center ΔC
-   Exit:
    -   Normal Δθ
    -   Center ΔC
-   Cumulative drift (exit)
-   Final closure gap (when applicable)

### Optional (Nice-to-Have)

Small panel summary similar to measurement tool.

Console output remains primary.

------------------------------------------------------------------------

## 13. Validation Plan

Test on:

1.  Trefoil (currently failing closure)
    -   Identify first wafer where mismatch spikes
    -   Measure drift accumulation
2.  Simple circle or helix (expected stable)
    -   Regression check

Acceptance Criteria:

-   Analyzer steps through all non-degenerate wafers
-   Mismatch origin can be identified
-   Closure gap is reported numerically
-   No silent failures

------------------------------------------------------------------------

## 14. Explicit Non-Goals (Phase 1)

-   Naming enforcement
-   Automatic correction of ordering
-   Surface-to-surface mesh deviation metrics
-   Monte Carlo simulation

------------------------------------------------------------------------

End of file.
