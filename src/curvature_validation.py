"""
Curvature validation and automatic correction for curves used in wafer generation.

This module provides tools to detect when a curve is too tight for straight
cylindrical wafers to follow, and optionally correct the curve or warn the user.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any


def calculate_curve_curvature(points: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Calculate the curvature at each point along a curve.

    Uses a finite difference approximation with a sliding window for smoothing.

    Args:
        points: Array of shape (N, 3) containing [x, y, z] coordinates
        window: Number of points to use for smoothing (must be odd). Default 3.

    Returns:
        Array of curvature values at each point (1/radius of curvature)
    """
    if len(points) < 3:
        return np.zeros(len(points))

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Calculate first derivative (tangent vectors)
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # Tangent vectors (normalized)
    tangents = diffs / distances[:, np.newaxis]

    # Calculate second derivative (change in tangent direction)
    dtangents = np.diff(tangents, axis=0)

    # Curvature magnitude
    curvatures = np.linalg.norm(dtangents, axis=1) / distances[1:]

    # Pad to match original length
    curvatures = np.concatenate([[curvatures[0]], curvatures, [curvatures[-1]]])

    # Apply smoothing window if requested
    if window > 1:
        half_window = window // 2
        smoothed = np.zeros_like(curvatures)
        for i in range(len(curvatures)):
            start = max(0, i - half_window)
            end = min(len(curvatures), i + half_window + 1)
            smoothed[i] = np.mean(curvatures[start:end])
        curvatures = smoothed

    return curvatures


def validate_curve_for_cylinders(
        points: np.ndarray,
        cylinder_radius: float,
        min_radius_factor: float = 4.0,
        return_details: bool = False
) -> Dict[str, Any]:
    """
    Validate that a curve can be followed by straight cylindrical wafers.

    Args:
        points: Array of shape (N, 3) containing [x, y, z] coordinates
        cylinder_radius: Radius of the cylindrical wafers
        min_radius_factor: Minimum ratio of curve radius to cylinder radius.
            Default 4.0 means curve radius must be at least 4× cylinder radius.
        return_details: If True, include detailed curvature information

    Returns:
        Dictionary with validation results:
        - 'valid': bool - True if curve can be built
        - 'min_curve_radius': float - Minimum radius of curvature
        - 'max_curvature': float - Maximum curvature (1/min_radius)
        - 'max_curvature_location': int - Index where max curvature occurs
        - 'required_min_radius': float - Minimum radius required
        - 'safety_margin': float - How much margin we have (negative = too tight)
        - 'curvatures': np.ndarray - (if return_details) All curvature values
        - 'message': str - Human-readable status message
    """
    curvatures = calculate_curve_curvature(points)

    # Find maximum curvature (tightest bend)
    max_curvature = np.max(curvatures)
    max_curvature_idx = np.argmax(curvatures)

    # Convert to radius
    if max_curvature > 0:
        min_curve_radius = 1.0 / max_curvature
    else:
        min_curve_radius = float('inf')

    # Required minimum radius
    required_min_radius = cylinder_radius * min_radius_factor

    # Safety margin (positive = safe, negative = too tight)
    safety_margin = min_curve_radius - required_min_radius

    # Determine validity
    is_valid = safety_margin >= 0

    # Generate message
    if is_valid:
        message = (f"✓ Curve is buildable. Min radius {min_curve_radius:.2f} "
                   f"(safety margin: {safety_margin:.2f})")
    else:
        message = (f"✗ Curve too tight! Min radius {min_curve_radius:.2f} "
                   f"< required {required_min_radius:.2f} "
                   f"(deficit: {-safety_margin:.2f})")

    result = {
        'valid': is_valid,
        'min_curve_radius': min_curve_radius,
        'max_curvature': max_curvature,
        'max_curvature_location': max_curvature_idx,
        'required_min_radius': required_min_radius,
        'safety_margin': safety_margin,
        'message': message
    }

    if return_details:
        result['curvatures'] = curvatures

    return result


def suggest_curve_corrections(
        curve_type: str,
        parameters: Dict[str, Any],
        validation_result: Dict[str, Any],
        cylinder_radius: float
) -> Dict[str, Any]:
    """
    Suggest parameter adjustments to make a curve buildable.

    Args:
        curve_type: Type of curve ('sinusoidal', 'helical', 'spiral', etc.)
        parameters: Current curve parameters
        validation_result: Result from validate_curve_for_cylinders()
        cylinder_radius: Radius of cylindrical wafers

    Returns:
        Dictionary with suggestions:
        - 'corrections': Dict of parameter names and new values
        - 'explanation': List of strings explaining each correction
        - 'estimated_improvement': float - Expected safety margin after corrections
    """
    corrections = {}
    explanations = []

    deficit = -validation_result['safety_margin']
    required_improvement_ratio = (
            validation_result['required_min_radius'] /
            validation_result['min_curve_radius']
    )

    if curve_type == 'sinusoidal':
        # For sine waves, curvature is proportional to amplitude / length²
        # Reduce amplitude or increase length

        current_amp = parameters.get('amplitude', 5.0)
        current_length = parameters.get('length', 50.0)

        # Strategy 1: Reduce amplitude (easier)
        new_amplitude = current_amp / required_improvement_ratio
        corrections['amplitude'] = round(new_amplitude, 2)
        explanations.append(
            f"Reduce amplitude from {current_amp:.1f} to {new_amplitude:.1f} "
            f"(gentler curves)"
        )

        # Strategy 2: Increase length (alternative)
        new_length = current_length * math.sqrt(required_improvement_ratio)
        explanations.append(
            f"OR increase length from {current_length:.1f} to {new_length:.1f} "
            f"(stretches the wave)"
        )

    elif curve_type == 'helical':
        # For helices, reduce pitch or increase radius
        current_pitch = parameters.get('pitch', 2.5)
        current_radius = parameters.get('radius', 10.0)

        # Increasing radius reduces curvature
        new_radius = current_radius * required_improvement_ratio
        corrections['radius'] = round(new_radius, 2)
        explanations.append(
            f"Increase radius from {current_radius:.1f} to {new_radius:.1f}"
        )

    elif curve_type == 'spiral':
        # For spirals, the rate of radius change affects curvature
        current_max_r = parameters.get('max_radius', 10.0)
        current_min_r = parameters.get('min_radius', 5.0)
        current_turns = parameters.get('turns', 2.0)

        # Reduce rate of radius change
        new_turns = current_turns / required_improvement_ratio
        corrections['turns'] = round(new_turns, 2)
        explanations.append(
            f"Reduce turns from {current_turns:.1f} to {new_turns:.1f} "
            f"(slower spiral)"
        )

    else:
        # Generic advice
        explanations.append(
            f"Curve is too tight by factor of {required_improvement_ratio:.2f}. "
            f"Consider: reducing curvature, increasing scale, or using fewer cycles."
        )

    # Estimate improvement
    estimated_improvement = deficit * 0.8  # Conservative estimate

    return {
        'corrections': corrections,
        'explanation': explanations,
        'estimated_improvement': estimated_improvement
    }


def auto_correct_curve_parameters(
        curve_type: str,
        parameters: Dict[str, Any],
        cylinder_radius: float,
        min_radius_factor: float = 4.0,
        generator_func: callable = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Automatically adjust curve parameters to ensure buildability.

    This is an iterative approach that generates test curves and adjusts
    parameters until the curve is valid.

    Args:
        curve_type: Type of curve
        parameters: Current parameters
        cylinder_radius: Radius of cylindrical wafers
        min_radius_factor: Safety factor (curve_radius / cylinder_radius)
        generator_func: Function to generate curve from parameters

    Returns:
        Tuple of (corrected_parameters, validation_result)
    """
    if generator_func is None:
        raise ValueError("generator_func required for auto-correction")

    max_iterations = 10
    iteration = 0
    current_params = parameters.copy()

    while iteration < max_iterations:
        # Generate test curve
        test_points = np.array(generator_func(**current_params))

        # Validate
        result = validate_curve_for_cylinders(
            test_points, cylinder_radius, min_radius_factor
        )

        if result['valid']:
            return current_params, result

        # Get suggestions and apply most aggressive correction
        suggestions = suggest_curve_corrections(
            curve_type, current_params, result, cylinder_radius
        )

        if not suggestions['corrections']:
            break

        # Apply first correction
        for key, value in suggestions['corrections'].items():
            current_params[key] = value
            break  # Only apply one correction per iteration

        iteration += 1

    # If we get here, couldn't auto-correct
    return current_params, validate_curve_for_cylinders(
        np.array(generator_func(**current_params)),
        cylinder_radius,
        min_radius_factor
    )
