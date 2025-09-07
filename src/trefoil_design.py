import numpy as np


def generate_woodcut_trefoil(slices=180, **parameters):
    """
    Generate a trefoil curve optimized for wood cutting with cylindrical slices.

    Parameters:
        slices (int): Number of cutting slices (affects curve resolution)
        **parameters: Same parameters as original trefoil function

    Returns:
        np.ndarray of shape (slices, 3): Points optimized for woodcutting
    """
    R = float(parameters.get("major_radius", 6.0))
    r = float(parameters.get("tube_radius", 2.0))
    p = int(parameters.get("p", 2))
    q = int(parameters.get("q", 3))
    cx, cy, cz = parameters.get("center", (0.0, 0.0, 0.0))
    phase_deg = float(parameters.get("phase_deg", 0.0))
    scale_z = float(parameters.get("scale_z", 1.0))

    # Ensure we don't exceed 200 points
    n = min(slices, 200)

    # Generate parameter values with strategic spacing
    # Add slight randomization to avoid perfect regularity that can cause grain issues
    t0 = np.deg2rad(phase_deg)
    t_base = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)

    # Add small perturbations for more natural cutting (optional)
    jitter = parameters.get("jitter", 0.0)  # 0.01-0.03 recommended for natural variation
    if jitter > 0:
        t_perturbation = jitter * (np.random.random(n) - 0.5) * (2 * np.pi / n)
        t = t_base + t_perturbation + t0
    else:
        t = t_base + t0

    # Trefoil parametric equations with slight smoothing
    # Use slightly higher precision for manufacturing
    cq = np.cos(q * t)
    sq = np.sin(q * t)
    cp = np.cos(p * t)
    sp = np.sin(p * t)

    # Apply optional smoothing factor for easier cutting
    smooth_factor = parameters.get("smooth_factor", 1.0)  # 0.9-0.95 for gentler curves

    x = (R + r * cq * smooth_factor) * cp + cx
    y = (R + r * cq * smooth_factor) * sp + cy
    z = (r * sq * smooth_factor) * scale_z + cz

    pts = np.stack((x, y, z), axis=1).astype(float)

    # Optional: ensure points are well-distributed for cutting
    if parameters.get("optimize_spacing", True):
        pts = _optimize_cutting_spacing(pts)

    return pts


def _optimize_cutting_spacing(points):
    """
    Redistribute points to ensure more uniform spacing for cutting operations.
    """
    # Calculate cumulative arc length
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(distances)])

    # Create evenly spaced arc length samples
    total_length = cumulative_length[-1]
    target_lengths = np.linspace(0, total_length, len(points))

    # Interpolate to get evenly spaced points
    x_interp = np.interp(target_lengths, cumulative_length, points[:, 0])
    y_interp = np.interp(target_lengths, cumulative_length, points[:, 1])
    z_interp = np.interp(target_lengths, cumulative_length, points[:, 2])

    return np.column_stack([x_interp, y_interp, z_interp])


def analyze_cutting_requirements(points):
    """
    Analyze the curve to provide cutting guidance.
    """
    # Calculate chord information for each slice
    center = np.mean(points, axis=0)
    radii = np.sqrt(np.sum((points - center) ** 2, axis=1))

    # Z-height variation (important for stacking slices)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_range = z_max - z_min

    # Maximum radius for cylinder stock
    max_radius = np.max(radii)

    # Angle between consecutive cuts
    n_points = len(points)
    angle_per_slice = 360.0 / n_points

    print(f"Cutting Analysis:")
    print(f"  Number of slices: {n_points}")
    print(f"  Cylinder diameter needed: {2 * max_radius:.2f}")
    print(f"  Height range: {z_range:.2f} (from {z_min:.2f} to {z_max:.2f})")
    print(f"  Angle per slice: {angle_per_slice:.2f}°")
    print(f"  Average radius: {np.mean(radii):.2f}")

    return {
        'n_slices': n_points,
        'cylinder_diameter': 2 * max_radius,
        'height_range': z_range,
        'angle_per_slice': angle_per_slice,
        'radii': radii
    }


# Example usage for woodcutting
if __name__ == "__main__":
    # Generate curve optimized for wood cutting
    trefoil_points = generate_woodcut_trefoil(
        slices=150,  # Good balance of detail vs. cutting time
        major_radius=8.0,  # Larger for easier handling
        tube_radius=2.5,
        smooth_factor=0.92,  # Slightly smoother for easier cutting
        jitter=0.02,  # Small natural variation
        optimize_spacing=True
    )

    # Analyze cutting requirements
    cutting_info = analyze_cutting_requirements(trefoil_points)

    # Optional: Save points for CAD/CAM software
    np.savetxt('/home/don/.plantbuilder/trefoil_points.csv', trefoil_points,
              delimiter=',', header='X,Y,Z', comments='')

    print(f"\nGenerated {len(trefoil_points)} points for cutting")
