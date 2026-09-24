import os
import re
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import matplotlib

# Use a non-GUI backend to prevent crashes on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import random

# Check if pytorch3d is installed, provide a helpful error message if not.
try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    print("Error: pytorch3d is not installed. Please install it to calculate Chamfer distance.")
    print(
        "Installation command: pip install 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable'")
    exit()


# --- Integrated from debug script ---
# ANSI color codes for clearer output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


# ---

# Define material and interaction types
MATERIAL_MAP = {
    '0': 'elastic', '1': 'elastic',
    '2': 'plastic', '3': 'plastic',
    '4': 'fluid', '5': 'fluid',
    '6': 'sand', '7': 'sand'
}
# CORRECTED: Changed 'p-f' to 'f-p' to match the script's sorting logic
INTERACTION_TYPES = ['e-p', 'e-f', 'e-s', 'f-p', 'p-s', 'f-s']


def get_interaction_type(scene_name):
    """Determines the interaction type from the scene folder name (e.g., '24')."""
    if not re.fullmatch(r'\d{2}', scene_name):
        return None, "Invalid folder name (must be two digits)"

    type1 = MATERIAL_MAP.get(scene_name[0])
    type2 = MATERIAL_MAP.get(scene_name[1])

    if not type1 or not type2:
        return None, "Invalid material digits in folder name"

    key = tuple(sorted((type1[0], type2[0])))
    interaction_key = f"{key[0]}-{key[1]}"

    if interaction_key not in INTERACTION_TYPES:
        return None, f"Unrecognized interaction type: {interaction_key}"

    return interaction_key, None


def load_gt_trajectory(gt_scene_dir: Path, num_frames: int = 30) -> list:
    """Loads and combines ground truth point clouds for two objects."""
    gt_point_clouds_dir = gt_scene_dir / "point_clouds"
    obj1_dir = gt_point_clouds_dir / "0"
    obj2_dir = gt_point_clouds_dir / "1"

    if not all([gt_point_clouds_dir.exists(), obj1_dir.exists(), obj2_dir.exists()]):
        return []

    combined_trajectory = []
    for i in range(num_frames):
        ply_path1 = obj1_dir / f"{i}.ply"
        ply_path2 = obj2_dir / f"{i}.ply"

        if not ply_path1.exists() or not ply_path2.exists():
            break

        try:
            pcd1 = o3d.io.read_point_cloud(str(ply_path1))
            pcd2 = o3d.io.read_point_cloud(str(ply_path2))

            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)

            combined_points = np.vstack((points1, points2))
            combined_trajectory.append(torch.from_numpy(combined_points).float().cuda())
        except Exception:
            return []

    return combined_trajectory


def calculate_chamfer_per_step(pred_traj, gt_traj):
    """
    Calculates Chamfer distance for each timestep, aligning with the specific
    downsampling, random sampling, and scaling method from the reference code.
    """
    num_common_frames = min(len(pred_traj), len(gt_traj))

    distances = []
    for i in range(num_common_frames):
        pcd0_original = pred_traj[i]  # Prediction (on GPU)
        pcd1_original = gt_traj[i]  # Ground truth (on GPU)

        pred_points = pcd0_original.shape[0]
        gt_points = pcd1_original.shape[0]

        # Uniform downsampling to match point counts
        if gt_points > pred_points:
            # Downsample GT. Using torch.linspace for GPU compatibility
            indices = torch.linspace(0, gt_points - 1, pred_points, device=pcd1_original.device).long()
            pcd1 = pcd1_original[indices]
            pcd0 = pcd0_original
        elif pred_points > gt_points:
            # Downsample Prediction
            indices = torch.linspace(0, pred_points - 1, gt_points, device=pcd0_original.device).long()
            pcd0 = pcd0_original[indices]
            pcd1 = pcd1_original
        else:
            pcd0 = pcd0_original
            pcd1 = pcd1_original

        # Determine sample size for evaluation
        n_sample = 8192
        current_num_points = pcd0.shape[0]
        if n_sample > current_num_points:
            n_sample = current_num_points

        # Random sampling for evaluation.
        sample_indices_0 = torch.randperm(current_num_points, device=pcd0.device)[:n_sample]
        sample_indices_1 = torch.randperm(current_num_points, device=pcd1.device)[:n_sample]

        pcd0_sampled = pcd0[sample_indices_0]
        pcd1_sampled = pcd1[sample_indices_1]

        # Add batch dimension for pytorch3d
        pcd0_batch = pcd0_sampled.unsqueeze(0)
        pcd1_batch = pcd1_sampled.unsqueeze(0)

        # Calculate scaled Chamfer distance
        dist, _ = chamfer_distance(pcd0_batch, pcd1_batch)
        scaled_dist = (dist * 1e3).item()
        distances.append(scaled_dist)

    return distances


def main(pred_dir_a, pred_dir_b, gt_dir, output_file):
    # --- ADDED: Diagnostic print to confirm path assignments ---
    print("\n" + "=" * 50)
    print("      Confirming Directory Assignments")
    print("=" * 50)
    print(f"  -> Path for L_CD_obj (blue, solid line):   {pred_dir_a}")
    print(f"  -> Path for L_CD_global (orange, dash): {pred_dir_b}")
    print("=" * 50 + "\n")
    # ---

    pred_path_a = Path(pred_dir_a)
    pred_path_b = Path(pred_dir_b)
    gt_path = Path(gt_dir)

    results_a = {key: [] for key in INTERACTION_TYPES}
    results_b = {key: [] for key in INTERACTION_TYPES}

    scene_folders = sorted([d for d in pred_path_a.iterdir() if d.is_dir()])

    processed_count = 0
    skipped_count = 0

    # --- ADDED: List of folders to manually bypass ---
    folders_to_bypass = ['04', '36', '57']
    print(f"{Colors.YELLOW}Bypassing the following folders as requested: {folders_to_bypass}{Colors.ENDC}")

    print("\n--- Starting Scene Validation and Processing ---")
    for scene_folder_a in tqdm(scene_folders, desc="Overall Progress"):
        scene_name = scene_folder_a.name
        status = f"Scene: {scene_name.ljust(5)}"

        # --- ADDED: Check if the scene should be bypassed ---
        if scene_name in folders_to_bypass:
            tqdm.write(f"{status} {Colors.YELLOW}[BYPASSED]{Colors.ENDC} Reason: Manually excluded by user.")
            skipped_count += 1
            continue

        interaction_key, error = get_interaction_type(scene_name)
        if error:
            tqdm.write(f"{status} {Colors.RED}[SKIPPED]{Colors.ENDC} Reason: {error}")
            skipped_count += 1
            continue

        # --- Check for all required files and folders ---
        scene_folder_b = pred_path_b / scene_name
        gt_scene_dir = gt_path / scene_name

        npy_files_a = list((scene_folder_a / "pred_traj").glob("*.npy"))
        npy_files_b = list((scene_folder_b / "pred_traj").glob("*.npy"))

        checks = {
            "Dir 'b' exists": scene_folder_b.is_dir(),
            "GT dir exists": gt_scene_dir.is_dir(),
            "Pred file in 'a'": bool(npy_files_a),
            "Pred file in 'b'": bool(npy_files_b),
        }

        failed_checks = [check for check, passed in checks.items() if not passed]
        if failed_checks:
            reason = ", ".join(failed_checks)
            tqdm.write(f"{status} {Colors.RED}[SKIPPED]{Colors.ENDC} Reason: Missing required files/folders ({reason})")
            skipped_count += 1
            continue

        # --- If all checks pass, proceed with loading and analysis ---
        tqdm.write(f"{status} {Colors.GREEN}[VALIDATED]{Colors.ENDC} Type: {interaction_key}. Processing...")

        try:
            pred_traj_a = [torch.from_numpy(frame).float().cuda() for frame in np.load(npy_files_a[0])]
            pred_traj_b = [torch.from_numpy(frame).float().cuda() for frame in np.load(npy_files_b[0])]
            gt_trajectory = load_gt_trajectory(gt_scene_dir, num_frames=len(pred_traj_a))
        except Exception as e:
            tqdm.write(f"{status} {Colors.RED}[SKIPPED]{Colors.ENDC} Reason: Error loading data file: {e}")
            skipped_count += 1
            continue

        if not gt_trajectory:
            tqdm.write(f"{status} {Colors.RED}[SKIPPED]{Colors.ENDC} Reason: Incomplete or invalid GT trajectory data.")
            skipped_count += 1
            continue

        chamfer_a = calculate_chamfer_per_step(pred_traj_a, gt_trajectory)
        chamfer_b = calculate_chamfer_per_step(pred_traj_b, gt_trajectory)

        # --- ADDED: Print per-frame CD values for debugging ---
        def format_cd_list(cd_list, chunk_size=10):
            lines = []
            for i in range(0, len(cd_list), chunk_size):
                chunk = cd_list[i:i + chunk_size]
                lines.append(" ".join([f"{val:7.4f}" for val in chunk]))
            return "\n      ".join(lines)

        if chamfer_a:
            tqdm.write(f"    -> CD (obj) per frame:\n      {format_cd_list(chamfer_a)}")
        if chamfer_b:
            tqdm.write(f"    -> CD (global) per frame:\n      {format_cd_list(chamfer_b)}")
        # ---

        if chamfer_a and chamfer_b:
            results_a[interaction_key].append(chamfer_a)
            results_b[interaction_key].append(chamfer_b)
            processed_count += 1
        else:
            skipped_count += 1
            tqdm.write(f"{status} {Colors.RED}[SKIPPED]{Colors.ENDC} Reason: Chamfer calculation failed.")

    print("\n" + "=" * 40)
    print("           PROCESSING SUMMARY")
    print("=" * 40)
    print(f"  Successfully processed {Colors.GREEN}{processed_count}{Colors.ENDC} scenes.")
    print(f"  Skipped {Colors.YELLOW}{skipped_count}{Colors.ENDC} scenes due to missing data or errors.")
    print("=" * 40 + "\n")

    # --- Averaging and Plotting ---
    def average_results(results):
        avg_results = {}
        for interaction_type, dist_list in results.items():
            if not dist_list: continue
            min_len = min(len(d) for d in dist_list)
            dist_array = np.array([d[:min_len] for d in dist_list])
            avg_results[interaction_type] = np.mean(dist_array, axis=0)
        return avg_results

    avg_results_a = average_results(results_a)
    avg_results_b = average_results(results_b)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle('Average Chamfer Distance per Timestep by Interaction Type', fontsize=16)

    title_bbox = dict(boxstyle='round,pad=0.3', fc='lightgray', ec='none', alpha=0.8)

    # Find the global maximum y-value across all datasets for a shared y-axis
    global_y_max = 0
    for interaction_type in INTERACTION_TYPES:
        if interaction_type in avg_results_a:
            global_y_max = max(global_y_max, np.max(avg_results_a[interaction_type]))
        if interaction_type in avg_results_b:
            global_y_max = max(global_y_max, np.max(avg_results_b[interaction_type]))

    y_limit = global_y_max * 1.15  # Add a 15% buffer to the top

    for i, interaction_type in enumerate(INTERACTION_TYPES):
        ax = axes[i]
        ax.set_title(interaction_type.upper(), loc='left', fontweight='bold', bbox=title_bbox)
        ax.set_xlabel("Timestep")
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set shared y-axis limit for all plots
        ax.set_ylim(0, y_limit)

        # Force every subplot to show its y-axis tick labels
        ax.tick_params(axis='y', labelleft=True)

        # Only set y-label for the leftmost plots to avoid clutter
        if i % 3 == 0:
            ax.set_ylabel("Chamfer Distance")

        if interaction_type in avg_results_a and interaction_type in avg_results_b:
            data_a = avg_results_a[interaction_type]
            data_b = avg_results_b[interaction_type]

            ax.plot(range(len(data_a)), data_a, marker='o', linestyle='-', color='C0', markersize=4,
                    label=r'$L_{CD}^{obj}$')
            ax.plot(range(len(data_b)), data_b, marker='^', linestyle='--', color='C1', markersize=4,
                    label=r'$L_{CD}^{global}$')
        else:
            ax.text(0.5, 0.5, 'No complete data for this category',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize='large')

    for j in range(len(INTERACTION_TYPES), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

    # Commented out to prevent crashes on servers
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare trajectory Chamfer distance from two sources.")
    parser.add_argument("--pred_dir_a", type=str, required=True,
                        help="Path to the first prediction directory 'a' (obj).")
    parser.add_argument("--pred_dir_b", type=str, required=True,
                        help="Path to the second prediction directory 'b' (global).")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the ground truth directory.")
    parser.add_argument("--output_file", type=str, default="chamfer_comparison.png",
                        help="Path to save the output plot.")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: This script requires a CUDA-enabled GPU.")
        exit()

    main(args.pred_dir_a, args.pred_dir_b, args.gt_dir, args.output_file)

