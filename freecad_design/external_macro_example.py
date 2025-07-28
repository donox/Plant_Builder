# Example external macro to move all segments
import FreeCAD
from your_module.flex_segment import FlexSegment

# Find all segments with stored transforms
segments = FlexSegment.find_segments_with_transforms()

# Apply additional transform to each segment
additional_transform = FreeCAD.Placement(FreeCAD.Vector(10, 0, 0), FreeCAD.Rotation())

for segment_lcs in segments:
    print(f"Moving segment: {segment_lcs.Label}")
    FlexSegment.apply_transform_to_segment_and_vertices(segment_lcs, additional_transform)

FreeCAD.ActiveDocument.recompute()