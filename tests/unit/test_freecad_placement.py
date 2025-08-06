import numpy as np
import FreeCAD

def test_freecad_placement():
    """Test if FreeCAD placements are being applied correctly."""

    print("\n=== Test 3: FreeCAD Placement Verification ===")

    # Test with actual wafer data from your system
    test_wafer_data = [
        {
            'start_pos': np.array([8.0, 0.0, 0.0]),
            'end_pos': np.array([7.4, 2.9, 0.24]),
            'expected_direction': 'Should point toward end_pos'
        },
        {
            'start_pos': np.array([4.6, 6.5, 0.6]),
            'end_pos': np.array([-2.6, 7.5, 1.2]),
            'expected_direction': 'Should follow curve tangent'
        }
    ]

    for i, wafer_data in enumerate(test_wafer_data):
        print(f"\nTest Wafer {i + 1}:")
        start_pos = wafer_data['start_pos']
        end_pos = wafer_data['end_pos']

        print(f"  Start: [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
        print(f"  End:   [{end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f}]")

        # Calculate expected wafer direction
        wafer_vector = end_pos - start_pos
        wafer_length = np.linalg.norm(wafer_vector)
        expected_direction = wafer_vector / wafer_length

        print(
            f"  Expected direction: [{expected_direction[0]:.3f}, {expected_direction[1]:.3f}, {expected_direction[2]:.3f}]")

        # Test your actual placement calculation
        try:
            # Copy your exact placement calculation here
            placement = FreeCAD.Placement(
                FreeCAD.Vector(start_pos[0], start_pos[1], start_pos[2]),
                FreeCAD.Rotation()  # Start with no rotation to test positioning
            )

            print(f"  FreeCAD Position: [{placement.Base.x:.3f}, {placement.Base.y:.3f}, {placement.Base.z:.3f}]")
            print(
                f"  FreeCAD Rotation: [{placement.Rotation.getYawPitchRoll()[0]:.1f}°, {placement.Rotation.getYawPitchRoll()[1]:.1f}°, {placement.Rotation.getYawPitchRoll()[2]:.1f}°]")

            # Check if positioning is correct
            pos_error = np.array([placement.Base.x - start_pos[0],
                                  placement.Base.y - start_pos[1],
                                  placement.Base.z - start_pos[2]])
            pos_error_magnitude = np.linalg.norm(pos_error)

            if pos_error_magnitude < 0.001:
                print(f"  ✅ Position correct")
            else:
                print(f"  ❌ Position error: {pos_error_magnitude:.6f}")

        except Exception as e:
            print(f"  ❌ Error in placement: {e}")

    def test_wafer_chain_continuity():
        """Test if wafers connect properly end-to-end."""

        print("\n=== Test 4: Wafer Chain Continuity ===")

        # Get actual wafer data from your system
        wafer_positions = []  # You'll need to extract this from your actual run

        # Example format:
        wafer_positions = [
            {'start': [8.0, 0.0, 0.0], 'end': [7.4, 2.9, 0.24]},
            {'start': [7.4, 2.9, 0.24], 'end': [4.6, 6.5, 0.6]},
            # ... add more from your actual debug output
        ]

        print(f"Testing {len(wafer_positions)} wafer connections:")

        for i in range(len(wafer_positions) - 1):
            current_wafer = wafer_positions[i]
            next_wafer = wafer_positions[i + 1]

            current_end = np.array(current_wafer['end'])
            next_start = np.array(next_wafer['start'])

            gap = np.linalg.norm(next_start - current_end)

            print(f"  Wafer {i + 1} → {i + 2}: Gap = {gap:.6f}")

            if gap < 0.001:
                print(f"    ✅ Good connection")
            else:
                print(f"    ❌ Gap detected - wafers not connecting")
                print(f"    Current end:  [{current_end[0]:.3f}, {current_end[1]:.3f}, {current_end[2]:.3f}]")
                print(f"    Next start:   [{next_start[0]:.3f}, {next_start[1]:.3f}, {next_start[2]:.3f}]")