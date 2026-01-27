"""
test_transform.py - Test LCS-based Part alignment

Run this to verify transformation math works correctly.
"""

import FreeCAD as App
import Part
from core.logging_setup import get_logger

logger = get_logger(__name__)


def run_transform_test():
    """Test that LCS-based alignment works as expected"""
    logger.info("=" * 60)
    logger.info("Starting transformation test")
    logger.info("=" * 60)

    # Clear and create test document
    if App.ActiveDocument:
        doc_name = App.ActiveDocument.Name
        App.closeDocument(doc_name)

    doc = App.newDocument("TransformTest")
    logger.info("Created test document")

    # ========== PART 1 (like Base segment) ==========
    part1 = doc.addObject("App::Part", "Part1_Base")
    # Random position for Part1
    part1.Placement = App.Placement(App.Vector(10, -5, 20), App.Rotation())
    logger.info(f"Created Part1 at random position: {part1.Placement.Base}")

    # Box1: Extends in Z direction (vertical), NO rotation
    box1 = doc.addObject("Part::Box", "Box1")
    box1.Length = 2   # X dimension
    box1.Width = 2    # Y dimension
    box1.Height = 5   # Z dimension - this is the extension direction
    # Center the box at origin in local space
    box1.Placement = App.Placement(App.Vector(-1, -1, 0), App.Rotation())
    part1.addObject(box1)
    logger.info(f"Created Box1: 2x2x5, extends in local +Z")

    # LCS at box1's EXIT (top face at Z=5)
    # Since box has no rotation, LCS is simple
    lcs1_exit = doc.addObject("PartDesign::CoordinateSystem", "LCS1_Exit")
    lcs1_exit.Placement = App.Placement(
        App.Vector(0, 0, 5),  # Top of box in local coords
        App.Rotation()        # No rotation - Z points in +Z
    )
    part1.addObject(lcs1_exit)
    logger.info(f"Created LCS1_Exit at local (0,0,5), Z-axis points in +Z")

    # ========== PART 2 (like Curve Left segment) ==========
    part2 = doc.addObject("App::Part", "Part2_Curve")
    # Random position for Part2 (different from Part1)
    part2.Placement = App.Placement(App.Vector(-15, 30, -10), App.Rotation())
    logger.info(f"Created Part2 at random position: {part2.Placement.Base}")

    # Box2: Extends in X direction (horizontal), rotated so it extends in X
    # Box2: I want it to extend in the +X direction in its local frame
    # Start with a box at origin, no rotation
    # Length=5 means it extends from (0,0,0) to (5,0,0) in +X
    # The entry face (where it connects) is at x=0, centered at (0, 0, 0)
    # The exit face is at x=5, centered at (5, 0, 0)

    box2 = doc.addObject("Part::Box", "Box2")
    box2.Length = 5  # Extends in X
    box2.Width = 2  # Y dimension
    box2.Height = 2  # Z dimension

    # Position the box: center it in Y and Z, start at X=0
    # Box origin is at one corner, so offset to center the YZ cross-section
    box2.Placement = App.Placement(
        App.Vector(0, -1, -1),  # Offset so box is centered in YZ
        App.Rotation()  # NO rotation - extends in +X
    )
    part2.addObject(box2)
    logger.info(f"Created Box2: extends from X=0 to X=5, centered in YZ")

    # LCS at entry face: at X=0, Y=0, Z=0 (center of entry face)
    # Z-axis of LCS points in +X direction (along box extension)
    lcs2_entry = doc.addObject("PartDesign::CoordinateSystem", "LCS2_Entry")
    lcs2_entry.Placement = App.Placement(
        App.Vector(0, 0, 0),  # Entry face center
        App.Rotation(App.Vector(0, 1, 0), 90)  # Rotate so Z points in +X
    )
    part2.addObject(lcs2_entry)
    logger.info(f"Created LCS2_Entry: at (0,0,0), Z-axis points in +X")
    part2.addObject(lcs2_entry)
    logger.info(f"Created LCS2_Entry at local (0,0,0), Z-axis points in +X (due to rotation)")

    doc.recompute()
    # ========== SHOW INITIAL STATE ==========
    logger.info("\n" + "=" * 60)
    logger.info("INITIAL STATE (before alignment)")
    logger.info("=" * 60)
    logger.info(f"Part1 placement: {part1.Placement}")
    logger.info(f"Part2 placement: {part2.Placement}")
    logger.info(f"Boxes are at RANDOM positions, not aligned")

    # ========== CALCULATE TRANSFORMATION ==========
    logger.info("\n" + "=" * 60)
    logger.info("CALCULATING ALIGNMENT")
    logger.info("=" * 60)

    # Get world coordinates of target (Part1's exit)
    part1_exit_world = part1.Placement.multiply(lcs1_exit.Placement)
    logger.info(f"Part1 exit LCS (world): {part1_exit_world}")

    # Get local coordinates of Part2's entry
    part2_entry_local = lcs2_entry.Placement
    logger.info(f"Part2 entry LCS (local): {part2_entry_local}")

    # Calculate adjusted placement: target * local_entry.inverse()
    adjusted_placement = part1_exit_world.multiply(part2_entry_local.inverse())
    logger.info(f"Calculated adjusted Part2 placement: {adjusted_placement}")
    doc.recompute()

    # Debug: Show actual box positions in world space
    logger.info("\n" + "=" * 60)
    logger.info("BOX GEOMETRY IN WORLD SPACE (before alignment)")
    logger.info("=" * 60)

    # Box1 in world coords
    box1_world = part1.Placement.multiply(box1.Placement)
    logger.info(f"Box1 placement (world): {box1_world}")
    logger.info(f"Box1 extends from Z=0 to Z=5 in its local frame")

    # Box2 in world coords
    box2_world = part2.Placement.multiply(box2.Placement)
    logger.info(f"Box2 placement (world): {box2_world}")
    logger.info(f"Box2 extends from X=0 to X=5 in its local frame")

    # ========== APPLY TRANSFORMATION ==========
    logger.info("\nApplying transformation to Part2...")
    part2.Placement = adjusted_placement
    doc.recompute()
    logger.info("Transformation applied")

    # ========== VERIFY ALIGNMENT ==========
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    part2_entry_world = part2.Placement.multiply(part2_entry_local)
    logger.info(f"Part2 entry LCS (world after): {part2_entry_world}")
    logger.info(f"Should match Part1 exit LCS:   {part1_exit_world}")

    # Check if they match
    pos_match = (part2_entry_world.Base - part1_exit_world.Base).Length < 1e-6
    rot_match = part2_entry_world.Rotation.isSame(part1_exit_world.Rotation, 1e-6)

    logger.info(f"\nPosition match: {pos_match}")
    logger.info(f"Rotation match: {rot_match}")

    if pos_match and rot_match:
        logger.info("\n✓ SUCCESS: LCS are aligned!")
        logger.info("✓ Boxes should be visually connected end-to-end")
        logger.info("✓ Box2 should be at top of Box1, extending horizontally")
    else:
        logger.warning("\n✗ FAILED: LCS do not align")
        logger.warning(f"Position error: {(part2_entry_world.Base - part1_exit_world.Base).Length}")

    logger.info("\n" + "=" * 60)
    logger.info("Check FreeCAD GUI:")
    logger.info("- Part1 (vertical box) should be at one random location")
    logger.info("- Part2 (horizontal box) should connect to top of Part1")
    logger.info("- They should form an L-shape")
    logger.info("=" * 60)

    return doc