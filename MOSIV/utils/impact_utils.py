"""
Impact Analysis Part 1: Determine deformation, impact velocity, contact area, and collision time
Accurate detection at 24 fps with sub-frame refinement
"""

import numpy as np
import trimesh as tm
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt


# ---------- Data structures ----------

@dataclass
class ImpactMetrics:
    """Container for impact analysis results"""
    # Timing
    t_hit: float  # Start of impact (sub-frame precision)
    t_end: float  # End of impact (sub-frame precision)
    t_max_compression: int  # Frame of maximum compression
    t_max_area: int  # Frame of maximum contact area (may differ from t_max_compression)
    contact_duration: float  # t_end - t_hit

    # Velocities
    v_pre_impact: np.ndarray  # Velocity just before impact [vx, vy, vz]
    v_post_impact: Optional[np.ndarray]  # Velocity after impact (if bounces)
    speed_pre: float  # Speed magnitude before impact
    speed_post: Optional[float]  # Speed magnitude after impact

    # Deformations
    L0: float  # Initial height (canonical)
    W0: float  # Initial width (canonical)
    delta_L_max: float  # Maximum axial deformation
    delta_W_max: float  # Maximum lateral deformation (at max compression frame)
    delta_W_at_max_area: float  # Lateral deformation at max area frame
    strain_axial: float  # delta_L_max / L0
    strain_lateral: float  # delta_W_max / W0

    # Contact
    A_max: float  # Maximum contact area
    A_at_max_compression: float  # Contact area at max compression frame
    contact_areas: List[float]  # Contact area time series

    # Impact type
    impact_type: str  # "bounce" or "stick"

    # Raw data for plotting
    positions: np.ndarray  # Center of mass positions
    velocities: np.ndarray  # Velocities
    heights: np.ndarray  # Heights (for deformation)
    widths: np.ndarray  # Widths (for deformation)
    frames: List[int]  # Frame indices


# ---------- Utilities ----------

def robust_extent(pc, axis, percentile_low=5, percentile_high=95):
    """
    Compute robust extent along an axis using percentiles
    Args:
        pc: Point cloud (N, 3)
        axis: 0 (x), 1 (y), or 2 (z)
        percentile_low: Lower percentile (default 5%)
        percentile_high: Upper percentile (default 95%)
    Returns:
        extent: High - low percentile value
        low: Low percentile value
        high: High percentile value
    """
    # Handle torch tensors
    import torch
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    else:
        pc = np.asarray(pc)

    values = pc[:, axis]
    low = np.percentile(values, percentile_low)
    high = np.percentile(values, percentile_high)
    return high - low, low, high


def fit_ellipse_pca(pc, plane='xz', percentile=95):
    """
    Fit ellipse to point cloud projection using PCA
    Args:
        pc: Point cloud (N, 3)
        plane: Projection plane ('xz' or 'xy')
        percentile: Percentile for outlier trimming
    Returns:
        major_axis: Length of major axis (2 * sqrt(major eigenvalue))
        minor_axis: Length of minor axis (2 * sqrt(minor eigenvalue))
        angle: Rotation angle of major axis (radians)
    """
    # Handle torch tensors
    import torch
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    else:
        pc = np.asarray(pc)

    # Project to plane
    if plane == 'xz':
        points_2d = pc[:, [0, 2]]  # X and Z
    elif plane == 'xy':
        points_2d = pc[:, [0, 1]]  # X and Y
    else:
        raise ValueError(f"Unknown plane: {plane}")

    # Trim outliers using percentile
    if percentile < 100:
        center = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        threshold = np.percentile(distances, percentile)
        mask = distances <= threshold
        points_2d = points_2d[mask]

    # Center the points
    center = np.mean(points_2d, axis=0)
    centered = points_2d - center

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute axis lengths (2 standard deviations ~ 95% of data)
    major_axis = 2 * 2 * np.sqrt(eigenvalues[0])  # 2 * (2 std dev)
    minor_axis = 2 * 2 * np.sqrt(eigenvalues[1])

    # Angle of major axis
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    return major_axis, minor_axis, angle


def robust_centroid(pc, pct=5):
    """Compute centroid after trimming outliers"""
    # Handle torch tensors
    import torch
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    else:
        pc = np.asarray(pc)

    # Check for empty point cloud
    if pc.size == 0 or len(pc) == 0:
        return np.array([0.0, 0.0, 0.0])  # Return origin for empty cloud

    # Check if point cloud has valid shape
    if pc.ndim != 2 or pc.shape[1] != 3:
        return np.array([0.0, 0.0, 0.0])  # Return origin for invalid shape

    lo = np.percentile(pc, pct, axis=0)
    hi = np.percentile(pc, 100 - pct, axis=0)
    mask = ((pc >= lo) & (pc <= hi)).all(axis=1)
    trimmed = pc[mask]
    if trimmed.size == 0:
        trimmed = pc
    return trimmed.mean(axis=0)


def moving_average(x, k=3):
    """Apply moving average smoothing"""
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    if x.ndim == 1:
        xpad = np.pad(x, pad, mode='edge')
        return np.convolve(xpad, np.ones(k)/k, mode='valid')
    else:
        xpad = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
        out = np.empty_like(x)
        for i in range(x.shape[1]):
            out[:, i] = np.convolve(xpad[:, i], np.ones(k)/k, mode='same')[pad:-pad]
        return out


def contact_area_grid(pc, y_tol=0.015, cell_size=0.015):
    """
    Compute contact area using grid rasterization
    Args:
        pc: Point cloud (N, 3) where Y is vertical
        y_tol: Tolerance for contact detection (m)
        cell_size: Grid cell size (m)
    Returns:
        area: Contact area (m^2)
        n_cells: Number of occupied cells
    """
    # Handle torch tensors
    import torch
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    else:
        pc = np.asarray(pc)

    if len(pc) == 0:
        return 0.0, 0

    # Find points near ground (Y is vertical, ground is at minimum Y)
    y_min = pc[:, 1].min()
    contact_mask = pc[:, 1] <= y_min + y_tol
    contact_points = pc[contact_mask]

    if len(contact_points) == 0:
        return 0.0, 0

    # Project to XZ plane (horizontal plane when Y is up)
    xz = contact_points[:, [0, 2]]  # X and Z coordinates
    gx = np.floor(xz[:, 0] / cell_size).astype(int)
    gz = np.floor(xz[:, 1] / cell_size).astype(int)

    # Count unique occupied cells
    occupied = set(zip(gx, gz))
    n_cells = len(occupied)
    area = n_cells * (cell_size ** 2)

    return area, n_cells


def quadratic_zero_crossing(y_vals, x_vals=None, ascending=False):
    """
    Find zero crossing with sub-sample precision using quadratic fit
    Args:
        y_vals: Values to find zero crossing in
        x_vals: Corresponding x values (default: indices)
        ascending: True if looking for negative->positive crossing
    Returns:
        x_zero: Interpolated x value at zero crossing
        found: Whether a crossing was found
    """
    if x_vals is None:
        x_vals = np.arange(len(y_vals))

    # Find sign changes
    signs = np.sign(y_vals)
    if ascending:
        # Looking for -1 to +1
        crossings = np.where((signs[:-1] < 0) & (signs[1:] > 0))[0]
    else:
        # Looking for +1 to -1
        crossings = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0]

    if len(crossings) == 0:
        return None, False

    # Take first crossing
    i = crossings[0]

    # Fit quadratic to 3 points around crossing
    if i > 0 and i < len(y_vals) - 2:
        # Use i-1, i, i+1, i+2 (4 points for robustness)
        idx = [i-1, i, i+1, i+2] if i < len(y_vals) - 2 else [i-1, i, i+1]
    elif i == 0:
        idx = [i, i+1, i+2] if i < len(y_vals) - 2 else [i, i+1]
    else:
        idx = [i-1, i, i+1]

    # Fit polynomial
    p = np.polyfit(x_vals[idx], y_vals[idx], 2)

    # Find roots
    roots = np.roots(p)

    # Filter real roots within the interval
    real_roots = roots[np.isreal(roots)].real
    valid_roots = real_roots[(real_roots >= x_vals[i]) & (real_roots <= x_vals[i+1])]

    if len(valid_roots) > 0:
        return valid_roots[0], True
    else:
        # Fallback to linear interpolation
        x_zero = x_vals[i] + (x_vals[i+1] - x_vals[i]) * (-y_vals[i] / (y_vals[i+1] - y_vals[i]))
        return x_zero, True


# ---------- Main analysis ----------

def analyze_impact(
    frames: List[np.ndarray],
    fps: float = 24.0,
    # Contact detection
    cell_size: float = 0.015,
    y_tol: float = 0.015,  # Y is vertical!
    A_th_cells: int = 3,  # Threshold in number of cells
    # Velocity thresholds
    v_th: float = 0.03,  # m/s threshold for bounce detection
    M_consecutive: int = 2,  # Consecutive frames for bounce
    # Stabilization for stick
    W_stable: int = 4,  # Window for stability check
    alpha_area: float = 0.1,  # Relative area change threshold (10%)
    delta_L_stable_pct: float = 0.01,  # Relative height change threshold (1% of L0)
    # Smoothing
    smooth_k: int = 3,
) -> Optional[ImpactMetrics]:
    """
    Analyze first impact from point cloud sequence

    Returns:
        ImpactMetrics object or None if no impact detected or error occurs
    """

    if len(frames) < 3:
        return None

    T = len(frames)
    time_array = np.arange(T) / fps

    # ---------- Extract features ----------

    try:
        # Compute centroids and extents
        positions = np.array([robust_centroid(f) for f in frames])
        heights = np.array([robust_extent(f, axis=1)[0] for f in frames])  # Y is vertical
    except Exception as e:
        print(f"Error computing positions/heights: {e}")
        return None

    # Use PCA for lateral dimensions (more robust than bounding box)
    lateral_dims = []
    for f in frames:
        try:
            major, minor, angle = fit_ellipse_pca(f, plane='xz', percentile=95)
            # Use average of major and minor axes as characteristic width
            lateral_dims.append((major + minor) / 2)
        except:
            # Fallback to bounding box if PCA fails
            lateral_dims.append(np.mean([robust_extent(f, axis=0)[0], robust_extent(f, axis=2)[0]]))
    widths = np.array(lateral_dims)

    # Compute velocities directly from raw positions (no smoothing)
    # Smoothing can mask important impact dynamics
    dt = 1.0 / fps
    velocities = np.zeros_like(positions)
    for i in range(T):
        if i == 0:
            velocities[i] = (positions[1] - positions[0]) / dt
        elif i == T - 1:
            velocities[i] = (positions[-1] - positions[-2]) / dt
        else:
            velocities[i] = (positions[i+1] - positions[i-1]) / (2 * dt)

    # Vertical velocity (Y is up in your setup)
    v_y = velocities[:, 1]

    # Compute contact areas
    areas = []
    n_cells_list = []
    for f in frames:
        area, n_cells = contact_area_grid(f, y_tol=y_tol, cell_size=cell_size)
        areas.append(area)
        n_cells_list.append(n_cells)
    areas = np.array(areas)
    n_cells_array = np.array(n_cells_list)

    # ---------- Detect impact start ----------

    # Find first frame with sufficient contact
    t_hit_frame = None
    for t in range(1, T):
        # Require consecutive frames to avoid noise
        if t < T - 1:
            if n_cells_array[t] >= A_th_cells and n_cells_array[t+1] >= A_th_cells:
                t_hit_frame = t
                break
        else:
            if n_cells_array[t] >= A_th_cells:
                t_hit_frame = t
                break

    if t_hit_frame is None:
        return None

    # Sub-frame refinement for impact start
    # Look for area crossing threshold
    if t_hit_frame > 0:
        area_threshold = A_th_cells * cell_size**2
        y_vals = areas[t_hit_frame-1:t_hit_frame+2] - area_threshold
        x_vals = time_array[t_hit_frame-1:t_hit_frame+2]
        t_hit_refined, found = quadratic_zero_crossing(y_vals, x_vals, ascending=True)
        if found:
            t_hit = t_hit_refined
        else:
            t_hit = time_array[t_hit_frame]
    else:
        t_hit = time_array[t_hit_frame]

    # ---------- Detect impact end ----------

    t_end_frame = None
    t_end = None
    impact_type = None

    # Check for bounce
    for t in range(t_hit_frame + 1, min(t_hit_frame + 50, T - M_consecutive)):
        # Check if velocity becomes positive and contact disappears
        if all(v_y[t:t+M_consecutive] > v_th) and all(n_cells_array[t:t+M_consecutive] < A_th_cells):
            t_end_frame = t
            impact_type = "bounce"

            # Sub-frame refinement for velocity zero crossing
            search_start = max(t_hit_frame, t - 3)
            y_vals = v_y[search_start:t+2]
            x_vals = time_array[search_start:t+2]
            t_v_zero, found_v = quadratic_zero_crossing(y_vals, x_vals, ascending=True)

            # Sub-frame refinement for area disappearing
            y_vals_area = areas[t-1:min(t+3, T)] - A_th_cells * cell_size**2
            x_vals_area = time_array[t-1:min(t+3, T)]
            t_area_zero, found_a = quadratic_zero_crossing(y_vals_area, x_vals_area, ascending=False)

            # Take the later of the two
            candidates = []
            if found_v and t_v_zero is not None:
                candidates.append(t_v_zero)
            if found_a and t_area_zero is not None:
                candidates.append(t_area_zero)

            if candidates:
                t_end = max(candidates)
            else:
                t_end = time_array[t_end_frame]
            break

    # If no bounce, check for sticking/settling
    if t_end_frame is None:
        # Get initial height for scaling the threshold
        L0 = heights[0]
        delta_L_stable = delta_L_stable_pct * L0  # Scale threshold with object size

        for t in range(t_hit_frame + W_stable, T):
            window = slice(t - W_stable, t)

            # Check stability conditions
            v_y_stable = np.abs(v_y[window]).max() < v_th

            # Area stability (relative change)
            area_changes = np.abs(np.diff(areas[window])) / (areas[window][:-1] + 1e-6)
            area_stable = area_changes.max() < alpha_area

            # Height stability (now relative to object size)
            height_changes = np.abs(np.diff(heights[window]))
            height_stable = height_changes.max() < delta_L_stable

            if v_y_stable and area_stable and height_stable:
                t_end_frame = t - W_stable // 2  # Use middle of stable window
                t_end = time_array[t_end_frame]
                impact_type = "stick"
                break

    # Fallback if no end detected
    if t_end_frame is None:
        t_end_frame = min(t_hit_frame + 8, T - 1)  # Fixed 5 frames (~0.21s at 24fps)
        t_end = time_array[t_end_frame]
        impact_type = "unknown"

    # t_end_frame = min(t_hit_frame + 8, T - 1)  # Fixed 5 frames (~0.21s at 24fps) for IMPACT
    # t_end = time_array[t_end_frame]

    # ---------- Find maximum compression and maximum area ----------

    # For contact area: search within impact window (contact-based)
    impact_window = slice(t_hit_frame, t_end_frame + 1)
    areas_impact = areas[impact_window]
    t_max_area_rel = np.argmax(areas_impact)
    t_max_area = t_hit_frame + t_max_area_rel

    # For deformation: search in extended window (deformation can continue after impact)
    # Use 8-10 frames to capture post-impact deformation/settling
    deformation_window_end = min(t_hit_frame + 10, T - 1)  # Extended window for deformation
    deformation_window = slice(t_hit_frame, deformation_window_end + 1)

    # Find frame of maximum compression (minimum height) in extended window
    heights_deformation = heights[deformation_window]
    t_max_compression_rel = np.argmin(heights_deformation)
    t_max_compression = t_hit_frame + t_max_compression_rel

    # ---------- Extract metrics ----------

    # Canonical dimensions (first frame)
    L0 = heights[0]
    W0 = widths[0]

    # Deformations at maximum compression
    delta_L_max = L0 - heights[t_max_compression]
    delta_W_max = widths[t_max_compression] - W0

    # Deformations at maximum area (for completeness)
    delta_W_at_max_area = widths[t_max_area] - W0

    # Strains
    strain_axial = delta_L_max / L0 if L0 > 0 else 0
    strain_lateral = delta_W_max / W0 if W0 > 0 else 0

    # Contact areas
    A_max = areas[t_max_area]  # Maximum contact area
    A_at_max_compression = areas[t_max_compression]  # Area at max compression

    # Impact velocities (use frame before contact for true pre-impact velocity)
    if t_hit_frame > 0:
        v_pre_impact = velocities[t_hit_frame - 1]  # Velocity just before contact
    else:
        v_pre_impact = velocities[t_hit_frame]  # Fallback if impact at first frame
    speed_pre = np.linalg.norm(v_pre_impact)

    if impact_type == "bounce" and t_end_frame < T - 1:
        v_post_impact = velocities[t_end_frame]
        speed_post = np.linalg.norm(v_post_impact)
    else:
        v_post_impact = None
        speed_post = None

    # Contact duration
    contact_duration = t_end - t_hit

    return ImpactMetrics(
        t_hit=t_hit,
        t_end=t_end,
        t_max_compression=t_max_compression,
        t_max_area=t_max_area,
        contact_duration=contact_duration,
        v_pre_impact=v_pre_impact,
        v_post_impact=v_post_impact,
        speed_pre=speed_pre,
        speed_post=speed_post,
        L0=L0,
        W0=W0,
        delta_L_max=delta_L_max,
        delta_W_max=delta_W_max,
        delta_W_at_max_area=delta_W_at_max_area,
        strain_axial=strain_axial,
        strain_lateral=strain_lateral,
        A_max=A_max,
        A_at_max_compression=A_at_max_compression,
        contact_areas=areas.tolist(),
        impact_type=impact_type,
        positions=positions,
        velocities=velocities,
        heights=heights,
        widths=widths,
        frames=list(range(T))
    )


# ---------- Visualization ----------

def plot_impact_analysis(metrics: ImpactMetrics, fps: float = 24.0, save_path: Optional[str] = None):
    """Generate comprehensive impact analysis plots"""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    time = np.array(metrics.frames) / fps

    # Highlight impact window
    impact_mask = (time >= metrics.t_hit) & (time <= metrics.t_end)

    # 1. Vertical position
    ax = axes[0, 0]
    ax.plot(time, metrics.positions[:, 1], 'b-', label='Y position')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red', label='Impact')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--', label='Max compression')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Vertical Position')
    ax.legend()
    ax.grid(True)

    # 2. Vertical velocity
    ax = axes[0, 1]
    ax.plot(time, metrics.velocities[:, 1], 'g-', label='Vy')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Vertical Velocity (Impact: {metrics.impact_type})')
    ax.legend()
    ax.grid(True)

    # 3. Contact area
    ax = axes[0, 2]
    ax.plot(time, metrics.contact_areas, 'r-', label='Contact Area')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--')
    ax.scatter([time[metrics.t_max_compression]], [metrics.A_max],
               color='red', s=50, zorder=5, label=f'Max: {metrics.A_max:.4f} m²')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Contact Area (m²)')
    ax.set_title('Contact Area')
    ax.legend()
    ax.grid(True)

    # 4. Height (axial dimension)
    ax = axes[1, 0]
    ax.plot(time, metrics.heights, 'b-', label='Height')
    ax.axhline(metrics.L0, color='k', linestyle='--', alpha=0.5, label=f'L0: {metrics.L0:.3f} m')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'Axial Dimension (ΔL_max: {metrics.delta_L_max:.3f} m)')
    ax.legend()
    ax.grid(True)

    # 5. Width (lateral dimension)
    ax = axes[1, 1]
    ax.plot(time, metrics.widths, 'b-', label='Width')
    ax.axhline(metrics.W0, color='k', linestyle='--', alpha=0.5, label=f'W0: {metrics.W0:.3f} m')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Width (m)')
    ax.set_title(f'Lateral Dimension (ΔW_max: {metrics.delta_W_max:.3f} m)')
    ax.legend()
    ax.grid(True)

    # 6. Strains
    ax = axes[1, 2]
    axial_strain = (metrics.L0 - metrics.heights) / metrics.L0
    lateral_strain = (metrics.widths - metrics.W0) / metrics.W0
    ax.plot(time, axial_strain * 100, 'r-', label='Axial')
    ax.plot(time, lateral_strain * 100, 'b-', label='Lateral')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.axvline(time[metrics.t_max_compression], color='green', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Strain (%)')
    ax.set_title(f'Strains (εa: {metrics.strain_axial:.1%}, εl: {metrics.strain_lateral:.1%})')
    ax.legend()
    ax.grid(True)

    # 7. Speed magnitude
    ax = axes[2, 0]
    speeds = np.linalg.norm(metrics.velocities, axis=1)
    ax.plot(time, speeds, 'k-')
    ax.axvspan(metrics.t_hit, metrics.t_end, alpha=0.2, color='red')
    ax.scatter([metrics.t_hit], [metrics.speed_pre], color='blue', s=50,
               label=f'Pre: {metrics.speed_pre:.3f} m/s')
    if metrics.speed_post is not None:
        ax.scatter([metrics.t_end], [metrics.speed_post], color='green', s=50,
                   label=f'Post: {metrics.speed_post:.3f} m/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Magnitude')
    ax.legend()
    ax.grid(True)

    # 8. XZ trajectory (top view)
    ax = axes[2, 1]
    ax.plot(metrics.positions[:, 0], metrics.positions[:, 2], 'b-', alpha=0.5)
    ax.scatter(metrics.positions[impact_mask, 0], metrics.positions[impact_mask, 2],
               c='red', s=10, label='Impact')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Top View (XZ)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    # 9. Summary text
    ax = axes[2, 2]
    ax.axis('off')
    summary = f"""Impact Analysis Summary:

Impact type: {metrics.impact_type}
Contact duration: {metrics.contact_duration:.3f} s
  Start: {metrics.t_hit:.3f} s
  End: {metrics.t_end:.3f} s

Velocities:
  Pre-impact: {metrics.speed_pre:.3f} m/s
  Post-impact: {metrics.speed_post:.3f if metrics.speed_post else 'N/A'} m/s
  CoR: {metrics.speed_post/metrics.speed_pre:.2f if metrics.speed_post else 'N/A'}

Max deformation (frame {metrics.t_max_compression}):
  Axial: {metrics.delta_L_max:.3f} m ({metrics.strain_axial:.1%})
  Lateral: {metrics.delta_W_max:.3f} m ({metrics.strain_lateral:.1%})
  Contact area: {metrics.A_max:.4f} m²
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Impact Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig


# ---------- File I/O ----------

def load_frames_from_trajectory(trajectory_path: str, max_frames: int = 50) -> List[np.ndarray]:
    """Load point cloud frames from gs_frame_XXXX.ply files"""
    frames = []
    for i in range(max_frames):
        ply_path = Path(trajectory_path) / f"gs_frame_{i:04d}.ply"
        if ply_path.exists():
            mesh = tm.load_mesh(str(ply_path))
            frames.append(np.array(mesh.vertices))
        else:
            break
    return frames


def save_metrics(metrics: ImpactMetrics, output_path: str):
    """Save metrics to JSON file"""
    # Convert to serializable format
    data = {
        "impact_type": metrics.impact_type,
        "timing": {
            "t_hit": metrics.t_hit,
            "t_end": metrics.t_end,
            "t_max_compression_frame": metrics.t_max_compression,
            "contact_duration": metrics.contact_duration
        },
        "velocities": {
            "v_pre_impact": metrics.v_pre_impact.tolist(),
            "v_post_impact": metrics.v_post_impact.tolist() if metrics.v_post_impact is not None else None,
            "speed_pre": metrics.speed_pre,
            "speed_post": metrics.speed_post
        },
        "deformations": {
            "L0": metrics.L0,
            "W0": metrics.W0,
            "delta_L_max": metrics.delta_L_max,
            "delta_W_max": metrics.delta_W_max,
            "strain_axial": metrics.strain_axial,
            "strain_lateral": metrics.strain_lateral
        },
        "contact": {
            "A_max": metrics.A_max,
            "A_max_cells": int(metrics.A_max / (0.015 ** 2))  # Convert to cell count
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Metrics saved to {output_path}")


# ---------- Pipeline Wrapper ----------

def analyze_impact_from_trajectory(frames_or_path, fps=24.0, return_raw=False):
    """
    Simple wrapper for impact analysis pipeline with error handling
    Can be called directly from train_dynamic_MO.py

    Args:
        frames_or_path: Either a list of numpy arrays (point clouds) or path to trajectory folder
        fps: Frame rate
        return_raw: If True, return full ImpactMetrics object; if False, return simplified dict

    Returns:
        Dictionary with key metrics needed for estimation:
        - v_impact: Impact velocity vector [vx, vy, vz] (m/s)
        - speed_impact: Impact speed magnitude (m/s)
        - t_contact: Contact duration (s)
        - delta_L_max: Maximum axial deformation (m)
        - delta_W_max: Maximum lateral deformation (m)
        - strain_axial: Axial strain (ratio)
        - strain_lateral: Lateral strain (ratio)
        - A_max: Maximum contact area (m^2)
        - impact_type: "bounce" or "stick"

        Or ImpactMetrics object if return_raw=True
        Returns None if analysis fails
    """
    try:
        # Load frames if path is provided
        if isinstance(frames_or_path, str):
            frames = load_frames_from_trajectory(frames_or_path)
        else:
            frames = frames_or_path

        # Validate frames
        if not frames or len(frames) < 3:
            print("Warning: Insufficient frames for impact analysis")
            return None

        # Check for empty or invalid frames
        for i, frame in enumerate(frames):
            if frame is None or (hasattr(frame, 'size') and frame.size == 0):
                print(f"Warning: Frame {i} is empty or invalid, skipping impact analysis")
                return None
            if hasattr(frame, 'shape') and (len(frame.shape) != 2 or frame.shape[1] != 3):
                print(f"Warning: Frame {i} has invalid shape {frame.shape}, expected (N, 3)")
                return None

        # Run analysis
        metrics = analyze_impact(frames, fps=fps)

        if metrics is None:
            print("Warning: No impact detected")
            return None

        if return_raw:
            return metrics

    except Exception as e:
        print(f"ERROR in impact analysis: {type(e).__name__}: {e}")
        print("Skipping impact analysis for this object")
        return None

    # Return simplified dictionary for estimation
    return {
        # Velocities
        'v_impact': metrics.v_pre_impact,  # [vx, vy, vz]
        'v_pre_impact': metrics.v_pre_impact,  # Also include with full name
        'speed_impact': metrics.speed_pre,
        'v_post': metrics.v_post_impact,
        'v_post_impact': metrics.v_post_impact,  # Also include with full name
        'speed_post': metrics.speed_post,

        # Timing
        't_contact': metrics.contact_duration,
        't_hit': metrics.t_hit,
        't_end': metrics.t_end,
        't_max_compression': metrics.t_max_compression,
        't_max_area': metrics.t_max_area,
        'contact_duration': metrics.contact_duration,  # Also include with full name

        # Deformations
        'delta_L_max': metrics.delta_L_max,
        'delta_W_max': metrics.delta_W_max,
        'delta_W_at_max_area': metrics.delta_W_at_max_area,
        'strain_axial': metrics.strain_axial,
        'strain_lateral': metrics.strain_lateral,

        # Contact
        'A_max': metrics.A_max,  # Maximum area (at t_max_area)
        'A_at_max_compression': metrics.A_at_max_compression,  # Area at max compression
        'contact_areas': metrics.contact_areas,  # Full time series

        # Type
        'impact_type': metrics.impact_type,

        # Dimensions
        'L0': metrics.L0,
        'W0': metrics.W0,

        # Raw data needed for parameter estimation
        'heights': metrics.heights,
        'widths': metrics.widths,
        'positions': metrics.positions,
        'velocities': metrics.velocities,
        'frames': metrics.frames
    }


def estimate_material_parameters(metrics, mass, material_type, fps=24.0, eps_floor=1e-6):
    """
    Estimate material parameters from impact metrics based on material type.

    Args:
        metrics: ImpactMetrics object or dict from analyze_impact_from_trajectory
        mass: Object mass in kg
        material_type: Material type (10=elastic, 12=von_mises/plastic)
        fps: Frame rate
        eps_floor: Minimum strain threshold

    Returns:
        Dictionary with estimated parameters:
        - E: Young's modulus (Pa)
        - nu: Poisson's ratio (-)
        - yield_stress: Yield stress (Pa) - only for plastic materials
        - diagnostics: Additional diagnostic information
    """
    import numpy as np

    # Handle both dict and ImpactMetrics object
    if isinstance(metrics, dict):
        # For dict, access directly
        metrics_dict = metrics

        # Create wrapper for attribute access
        class MetricsWrapper:
            def __init__(self, d):
                self.data = d

            def __getattr__(self, name):
                if name in self.data:
                    return self.data[name]
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        metrics = MetricsWrapper(metrics_dict)

    results = {}

    # Check for valid contact data
    if not hasattr(metrics, 'A_max') or metrics.A_max <= 0:
        return {'error': 'Invalid or missing contact area'}
    if not hasattr(metrics, 't_contact') or metrics.t_contact <= 0:
        return {'error': 'Invalid or missing contact duration'}

    # Get strains at maximum compression
    t_star = int(metrics.t_max_compression)
    L0, W0 = float(metrics.L0), float(metrics.W0)
    L_t = float(metrics.heights[t_star])
    W_t = float(metrics.widths[t_star])

    # Signed engineering strains
    eps_ax = (L_t - L0) / (L0 + 1e-12)  # Negative for compression
    eps_lat = (W_t - W0) / (W0 + 1e-12)  # Positive for expansion

    # Average deceleration and stress
    v_pre = metrics.v_pre_impact
    v_post = metrics.v_post_impact if hasattr(metrics, 'v_post_impact') and metrics.v_post_impact is not None else np.zeros(3)

    # Use normal (y) component for ground impact
    v_pre_n = float(v_pre[1]) if len(v_pre) > 1 else 0.0
    v_post_n = float(v_post[1]) if len(v_post) > 1 else 0.0

    delta_v_n = abs(v_post_n - v_pre_n)
    a_avg = delta_v_n / metrics.t_contact
    a_avg = delta_v_n / (1/24)
    F_avg = mass * a_avg

    # USE AREA AT MAXIMUM COMPRESSION, NOT MAXIMUM AREA
    A_at_compression = metrics.A_at_max_compression  # Area at the frame of max compression
    sigma_avg = F_avg / A_at_compression

    # Young's modulus from stress at max compression / strain at max compression
    E = sigma_avg / max(abs(eps_ax), eps_floor)
    results['E'] = E

    # Poisson's ratio (valid for both elastic and plastic)
    nu = np.nan
    if eps_ax < -eps_floor:  # Only valid if in compression
        nu = -eps_lat / eps_ax
        nu = max(0.0, min(0.49, nu))  # Clamp to physical range
    results['nu'] = nu if np.isfinite(nu) else 0.3  # Default if can't estimate

    # Material-specific estimates
    from simulator import MPMSimulator

    if material_type == MPMSimulator.von_mises:  # Plastic material - also estimate yield stress
        # Use AVERAGE deceleration during impact (more robust than peak)
        t_hit_frame = int(np.floor(metrics.t_hit * fps))
        t_end_frame = int(np.ceil(metrics.t_end * fps))

        # Calculate accelerations
        v_y = np.array(metrics.velocities)[:, 1]
        dt = 1.0 / fps
        a_y = np.zeros_like(v_y)
        a_y[1:] = (v_y[1:] - v_y[:-1]) / dt

        # Calculate MEAN deceleration in impact window (more robust than peak)
        if t_end_frame > t_hit_frame:
            impact_window = slice(max(1, t_hit_frame), min(len(a_y), t_end_frame + 1))
            # Use mean of absolute decelerations (negative accelerations)
            a_impact = a_y[impact_window]
            a_impact_negative = a_impact[a_impact < 0]  # Only decelerations
            if len(a_impact_negative) > 0:
                a_mean_impact = abs(float(np.mean(a_impact_negative)))
            else:
                # Fallback to overall mean if no clear deceleration
                a_mean_impact = abs(float(np.mean(a_impact)))

            # Also track peak for diagnostics
            idx_local = np.argmin(a_y[impact_window])
            idx_peak = max(1, t_hit_frame) + idx_local
            a_peak = abs(float(a_y[idx_peak]))

            # Use MEAN contact area in impact window (more robust)
            areas_in_impact = metrics.contact_areas[t_hit_frame:t_end_frame+1]
            A_mean = float(np.mean(areas_in_impact)) if len(areas_in_impact) > 0 else metrics.A_max
            A_max_in_window = float(np.max(areas_in_impact)) if len(areas_in_impact) > 0 else metrics.A_max

            if A_mean <= 0:
                # Fallback to maximum if needed
                A_mean = A_max_in_window

            # Yield stress using MEAN deceleration and MEAN area (most robust)
            sigma_yield = (mass * a_mean_impact) / A_mean

            # Reasonable range for soft plastics (1kPa to 10MPa)
            sigma_yield = max(1e3, min(1e7, sigma_yield))
            results['yield_stress'] = sigma_yield * np.sqrt(2.0/3.0)

            # Additional diagnostics
            results['diagnostics'] = {
                'stress_avg': sigma_avg,
                'a_avg': a_avg,
                'a_mean_impact': a_mean_impact,
                'a_peak': a_peak,
                'frame_peak_decel': idx_peak,
                'A_mean': A_mean,
                'A_max_in_window': A_max_in_window,
                'A_max': metrics.A_max,
                'A_at_compression': A_at_compression
            }

    elif material_type == MPMSimulator.elasticity:  # Elastic material - E and nu only
        # Additional diagnostics for elastic
        results['diagnostics'] = {
            'stress_avg': sigma_avg,
            'a_avg': a_avg,
            'strain_axial': eps_ax,
            'strain_lateral': eps_lat
        }

    return results


def analyze_multi_object_impacts(object_frames_dict, fps=24.0):
    """
    Analyze impacts for multiple objects

    Args:
        object_frames_dict: Dictionary mapping object_id to list of point cloud frames
        fps: Frame rate

    Returns:
        Dictionary mapping object_id to impact metrics dict
    """
    results = {}

    for obj_id, frames in object_frames_dict.items():
        if obj_id == 0:  # Skip background
            continue

        print(f"Analyzing object {obj_id}...")
        metrics = analyze_impact_from_trajectory(frames, fps=fps, return_raw=False)

        if metrics is not None:
            results[obj_id] = metrics
            print(f"  Object {obj_id}: v={metrics['speed_impact']:.3f} m/s, "
                  f"t_c={metrics['t_contact']:.3f} s, A={metrics['A_max']:.4f} m^2")
        else:
            print(f"  Object {obj_id}: No impact detected")

    return results


# ---------- Main ----------

def main():
    """Example usage and testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze impact from GS trajectory")
    parser.add_argument("trajectory_path", help="Path to trajectory folder with gs_frame_XXXX.ply files")
    parser.add_argument("--fps", type=float, default=24.0, help="Frame rate")
    parser.add_argument("--output", type=str, help="Output path for metrics JSON")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # Load frames
    print(f"Loading frames from {args.trajectory_path}...")
    frames = load_frames_from_trajectory(args.trajectory_path)
    print(f"Loaded {len(frames)} frames")

    if len(frames) < 3:
        print("Error: Need at least 3 frames for analysis")
        return

    # Analyze impact
    print("Analyzing impact...")
    metrics = analyze_impact(frames, fps=args.fps)

    if metrics is None:
        print("No impact detected")
        return

    # Print results
    print(f"\n=== Impact Analysis Results ===")
    print(f"Impact type: {metrics.impact_type}")
    print(f"Contact duration: {metrics.contact_duration:.3f} s ({metrics.t_hit:.3f} to {metrics.t_end:.3f})")
    print(f"Pre-impact speed: {metrics.speed_pre:.3f} m/s")
    if metrics.speed_post is not None:
        print(f"Post-impact speed: {metrics.speed_post:.3f} m/s")
        print(f"Coefficient of restitution: {metrics.speed_post/metrics.speed_pre:.3f}")
    print(f"\nMaximum deformation:")
    print(f"  Axial: {metrics.delta_L_max:.3f} m ({metrics.strain_axial:.1%})")
    print(f"  Lateral: {metrics.delta_W_max:.3f} m ({metrics.strain_lateral:.1%})")
    print(f"  Contact area: {metrics.A_max:.4f} m² ({int(metrics.A_max / 0.015**2)} cells)")

    # Save metrics
    if args.output:
        save_metrics(metrics, args.output)

    # Generate plots
    if args.plot:
        plot_path = args.output.replace('.json', '_plot.png') if args.output else 'impact_analysis.png'
        plot_impact_analysis(metrics, fps=args.fps, save_path=plot_path)
        plt.show()


if __name__ == "__main__":
    main()