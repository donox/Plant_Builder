"""Example usage of reconstruction workflow."""

import FreeCAD
from reconstruction.reconstruction_workflow import ReconstructionWorkflow

# Initialize
doc = FreeCAD.ActiveDocument
if doc is None:
    doc = FreeCAD.newDocument("Reconstruction")

workflow = ReconstructionWorkflow(doc)

# Example 1: Reconstruct wafers from cut list
print("Example 1: Reconstructing wafers...")
result = workflow.reconstruct_from_cutlist(
    cutlist_file="output/cutting_list_spiral.txt",
    cylinder_diameter=2.0,  # 2 units diameter
    create_vertices_only=False,
    fuse_wafers=True
)

print(f"Reconstructed {len(result.segments)} segment(s)")
for segment in result.segments:
    print(f"  - {segment}")

# Example 2: Create only vertices for comparison
print("\nExample 2: Creating vertices only...")
result_vertices = workflow.reconstruct_from_cutlist(
    cutlist_file="output/cutting_list_spiral.txt",
    cylinder_diameter=2.0,
    create_vertices_only=True
)

# Example 3: Full validation
print("\nExample 3: Full validation...")
original_curve = {
    'type': 'spiral',
    'parameters': {
        'max_radius': 10.0,
        'max_height': 10.0,
        'turns': 3.0,
        'points': 100,
        'plane': 'xy'
    }
}

validation = workflow.validate_against_original(
    cutlist_file="output/cutting_list_spiral.txt",
    original_curve_spec=original_curve,
    cylinder_diameter=2.0,
    output_report="output/validation_report.txt"
)

print(f"\nValidation: {validation}")
print(f"  Max deviation: {validation.max_deviation:.4f}")
print(f"  RMS deviation: {validation.rms_deviation:.4f}")
print(f"  Status: {'PASSED' if validation.passed else 'FAILED'}")

doc.recompute()
print("\nReconstruction complete!")