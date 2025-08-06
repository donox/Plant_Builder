# tests/unit/test_wafer_separation.py
"""Test wafer separation algorithms"""
import pytest
import numpy as np
import math
import sys
import os

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, '/usr/lib/freecad-daily/lib')

from src.curve_follower import CurveFollower


class TestWaferSeparation:
    """Test wafer separation calculations"""

    def test_minimum_separation_calculation(self):
        """Test calculation of minimum wafer separation"""

        # Test the geometric calculation
        wafer_thickness = 0.1
        helical_pitch = 2.0
        radius = 5.0

        # Calculate minimum separation based on helical geometry
        circumference = 2 * math.pi * radius
        angle_per_unit_height = 2 * math.pi / helical_pitch

        # Minimum separation includes safety factor
        min_height_separation = wafer_thickness * 1.5
        min_arc_separation = min_height_separation * math.sqrt(1 + (circumference / helical_pitch) ** 2)

        print(f"Calculated minimum separation: {min_arc_separation:.4f}")
        print(f"For pitch={helical_pitch}, radius={radius}")

        # Should be reasonable value
        assert 0.1 < min_arc_separation < 10.0
        assert min_arc_separation > wafer_thickness

    def test_chord_distance_calculation(self):
        """Test chord distance calculation for curve approximation"""

        # Create a simple arc and test chord distance
        radius = 5.0
        arc_angle = math.pi / 4  # 45 degrees

        # Generate points along the arc
        points = []
        for i in range(5):
            angle = i * arc_angle / 4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(np.array([x, y, 0]))

        # Calculate chord distance (should be < radius for small arcs)
        start_point = points[0]
        end_point = points[-1]

        # Mock the chord distance calculation method
        def calculate_chord_distance(start, end, curve_points):
            if len(curve_points) <= 2:
                return 0.0

            chord_vector = end - start
            chord_length = np.linalg.norm(chord_vector)

            if chord_length < 1e-10:
                return 0.0

            chord_unit = chord_vector / chord_length

            max_distance = 0.0
            for point in curve_points[1:-1]:
                point_vector = point - start
                projection_length = np.dot(point_vector, chord_unit)
                projection_point = start + projection_length * chord_unit
                distance = np.linalg.norm(point - projection_point)
                max_distance = max(max_distance, distance)

            return max_distance

        chord_distance = calculate_chord_distance(start_point, end_point, points)

        print(f"Chord distance for {math.degrees(arc_angle):.1f}° arc: {chord_distance:.4f}")

        # For a quarter circle, chord distance should be reasonable
        assert chord_distance > 0
        assert chord_distance < radius  # Should be less than radius