# tests/unit/test_problematic_geometry.py
"""Reproduce the exact geometric problem from the logs"""
import numpy as np
import math


class TestProblematicGeometry:
    """Test the specific geometry that's causing extreme angles"""

    def test_sharp_curve_transitions(self):
        """Test what happens with sharp transitions in helical curves"""

        # Reproduce points from your log that cause problems
        problematic_points = [
            np.array([7.855, 1.514, 0.121]),  # Point 1 from log
            np.array([7.427, 2.973, 0.242]),  # Point 2 from log
            np.array([6.730, 4.325, 0.364]),  # Point 3 from log
        ]

        # Calculate tangent vectors like your code does
        def get_tangent(points, index):
            if index == 0:
                return points[1] - points[0]
            elif index == len(points) - 1:
                return points[-1] - points[-2]
            else:
                return points[index + 1] - points[index - 1]

        # Test the problematic calculation
        for i in range(len(problematic_points) - 1):
            start_point = problematic_points[i]
            end_point = problematic_points[i + 1]

            start_tangent = get_tangent(problematic_points, i)
            end_tangent = get_tangent(problematic_points, i + 1)

            chord_vector = end_point - start_point
            chord_unit = chord_vector / np.linalg.norm(chord_vector)

            # This is the calculation producing extreme angles
            dot_product_start = np.dot(start_tangent, chord_unit)
            dot_product_end = np.dot(end_tangent, chord_unit)

            start_angle = math.acos(np.clip(abs(dot_product_start), 0, 1))
            end_angle = math.acos(np.clip(abs(dot_product_end), 0, 1))

            print(f"Segment {i} → {i + 1}:")
            print(f"  Dot products: start={dot_product_start:.3f}, end={dot_product_end:.3f}")
            print(f"  Angles: start={math.degrees(start_angle):6.1f}°, end={math.degrees(end_angle):6.1f}°")

            # Identify why we're getting extreme angles
            if abs(dot_product_start) > 0.95 or abs(dot_product_end) > 0.95:
                print(f"  ⚠️  EXTREME: Tangent nearly parallel/antiparallel to chord")

            # Test normalization with these extreme values
            normalized_start = ((start_angle % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi) - math.pi
            normalized_end = ((end_angle % (2 * math.pi)) + 2 * math.pi) % (2 * math.pi) - math.pi

            print(
                f"  Normalized: start={math.degrees(normalized_start):6.1f}°, end={math.degrees(normalized_end):6.1f}°")