import json
import os
from pathlib import Path
import numpy as np

# Material mapping from metadata to MPM code
MATERIAL_MAP = {
    "elastic": 10,
    "elastoplastic": 12,  # von Mises
    "snow": 13,  # Drucker-Prager
    "sand": 13,  # Drucker-Prager
    "fluid": 11,  # Newtonian fluid
    "liquid": 11  # Newtonian fluid (same as fluid)
}

# Default physics parameters for each material type (values in actual scale, will be log10 in estimator)
DEFAULT_PARAMS = {
    "elastic": {
        "init_E": 1e5,  # Will be log10(1e5) = 5.0 in estimator
        "init_nu": 0.25,
        "trainable": ["Youngs modulus", "Poisson ratio"]
    },
    "elastoplastic": {
        "init_E": 1e4,  # Will be log10(1e4) = 4.0 in estimator (like PACNeRF)
        "init_nu": 0.25,
        "init_yield_stress": 1e3,  # Will be log10(1e3) = 3.0 in estimator (like PACNeRF)
        "trainable": ["Youngs modulus", "Poisson ratio", "Yield stress"]
    },
    "snow": {
        "init_E": 1e6,  # Will be log10(1e6) = 6.0 in estimator
        "init_nu": 0.3,
        "init_friction_alpha": 15.0,  # Not in log scale
        "trainable": ["friction angle"]  # Only friction angle, E/nu fixed
    },
    "sand": {
        "init_E": 1e6,  # Will be log10(1e6) = 6.0 in estimator (like PACNeRF)
        "init_nu": 0.3,
        "init_friction_alpha": 10.0,  # Like PACNeRF (not in log scale)
        "trainable": ["friction angle"]  # Only friction angle
    },
    "fluid": {
        "mu": 10.0,  # Will be log10(10) = 1.0 in estimator
        "kappa": 1e4,  # Will be log10(1e4) = 4.0 in estimator
        "trainable": ["kappa", "mu"]
    },
    "liquid": {
        "mu": 10.0,  # Will be log10(10) = 1.0 in estimator
        "kappa": 1e4,  # Will be log10(1e4) = 4.0 in estimator
        "trainable": ["kappa", "mu"]
    }
}

# Training parameters for each parameter type
TRAINING_PARAMS = {
    "Youngs modulus": {
        "lr_decay": True,
        "init_lr": 0.05,
        "final_lr": 0.01,
        "max_steps": 200
    },
    "Poisson ratio": {
        "lr_decay": True,
        "init_lr": 0.01,
        "final_lr": 0.005,
        "max_steps": 200
    },
    "Yield stress": {
        "lr_decay": True,
        "init_lr": 0.1,
        "final_lr": 0.01,
        "max_steps": 200
    },
    "friction angle": {
        "lr_decay": False,
        "init_lr": 1.0,
        "final_lr": 0.1,
        "max_steps": 100
    },
    "plastic viscosity": {
        "lr_decay": False,
        "init_lr": 0.01,
        "final_lr": 0.005,
        "max_steps": 100
    },
    "mu": {
        "lr_decay": False,
        "init_lr": 0.15,
        "final_lr": 0.01,
        "max_steps": 100
    },
    "kappa": {
        "lr_decay": False,
        "init_lr": 0.2,
        "final_lr": 0.1,
        "max_steps": 100
    }
}

def get_bbox_from_pointclouds(dataset_id):
    """Extract per-object bounding boxes from point cloud files"""
    import plyfile

    # Path to point cloud directory
    pc_dir = Path(f"/mnt/e/GenesisMO/{dataset_id}/point_clouds")

    if not pc_dir.exists():
        print(f"Warning: Point cloud directory not found for {dataset_id}, using default bounds")
        # Fallback to default global bounds
        margin = 0.6
        xyz_min = [-margin, 0.2, -margin]
        xyz_max = [margin, 0.65, margin]
        return xyz_min, xyz_max, None, None, None, None

    # Read first frame (0.ply) for each object
    obj1_ply_path = pc_dir / "0" / "0.ply"
    obj2_ply_path = pc_dir / "1" / "0.ply"

    try:
        # Read object 1 point cloud
        plydata1 = plyfile.PlyData.read(obj1_ply_path)
        vertices1 = plydata1['vertex']
        obj1_points = np.vstack([vertices1['x'], vertices1['y'], vertices1['z']]).T
        obj1_xyz_min = np.min(obj1_points, axis=0).tolist()
        obj1_xyz_max = np.max(obj1_points, axis=0).tolist()

        # Read object 2 point cloud
        plydata2 = plyfile.PlyData.read(obj2_ply_path)
        vertices2 = plydata2['vertex']
        obj2_points = np.vstack([vertices2['x'], vertices2['y'], vertices2['z']]).T
        obj2_xyz_min = np.min(obj2_points, axis=0).tolist()
        obj2_xyz_max = np.max(obj2_points, axis=0).tolist()

        # Also compute global bounds
        all_points = np.vstack([obj1_points, obj2_points])
        global_xyz_min = np.min(all_points, axis=0).tolist()
        global_xyz_max = np.max(all_points, axis=0).tolist()

        print(f"Dataset {dataset_id} bounds:")
        print(f"  Object 1: {obj1_xyz_min} to {obj1_xyz_max}")
        print(f"  Object 2: {obj2_xyz_min} to {obj2_xyz_max}")
        print(f"  Global:   {global_xyz_min} to {global_xyz_max}")

        return global_xyz_min, global_xyz_max, obj1_xyz_min, obj1_xyz_max, obj2_xyz_min, obj2_xyz_max

    except Exception as e:
        print(f"Error reading point clouds for {dataset_id}: {e}")
        # Fallback to default bounds
        margin = 0.6
        xyz_min = [-margin, 0.2, -margin]
        xyz_max = [margin, 0.65, margin]
        return xyz_min, xyz_max, None, None, None, None

def create_config(dataset_id, metadata_path):
    """Create config for a single dataset"""

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get object materials
    obj1_material = metadata["obj1"]["material"]
    obj2_material = metadata["obj2"]["material"]

    # Get bounding boxes from actual point cloud data
    global_xyz_min, global_xyz_max, obj1_xyz_min, obj1_xyz_max, obj2_xyz_min, obj2_xyz_max = get_bbox_from_pointclouds(dataset_id)

    # Determine voxel size and density grid size based on materials (from PACNeRF)
    has_fluid = obj1_material in ["fluid", "liquid"] or obj2_material in ["fluid", "liquid"]
    has_sand_snow = obj1_material in ["sand", "snow"] or obj2_material in ["sand", "snow"]
    has_plastic = obj1_material == "elastoplastic" or obj2_material == "elastoplastic"

    if has_sand_snow:
        voxel_size = 0.015  # Finer for granular materials
        density_grid_size = 0.12
        density_min_th = 0.67
        density_max_th = 0.8
    elif has_fluid:
        voxel_size = 0.02  # PACNeRF uses 0.02 for fluid
        density_grid_size = 0.12
        density_min_th = 0.67
        density_max_th = 0.9
    elif has_plastic:
        voxel_size = 0.02
        density_grid_size = 0.1
        density_min_th = 0.7
        density_max_th = 0.9
    else:  # elastic
        voxel_size = 0.02
        density_grid_size = 0.1
        density_min_th = 0.5
        density_max_th = 0.7

    # Create sub_objects
    sub_objects = []

    # Object 1
    obj1_params = DEFAULT_PARAMS[obj1_material].copy()
    obj1_trainable = obj1_params.pop("trainable")

    # Use rho from metadata
    obj1_rho = metadata["obj1"].get("rho", 1000.0)

    sub_obj1 = {
        "name": metadata["obj1"]["geometry"],
        "object_id": 1,
        "material": MATERIAL_MAP[obj1_material],
        **obj1_params,
        "rho": obj1_rho,  # Use rho from metadata
        "init_vel": [0.0, 0.0, 0.0],  # Set to zero as requested
        "color": metadata["obj1"]["surface_color"]
    }
    sub_objects.append(sub_obj1)

    # Object 2
    obj2_params = DEFAULT_PARAMS[obj2_material].copy()
    obj2_trainable = obj2_params.pop("trainable")

    # Use rho from metadata
    obj2_rho = metadata["obj2"].get("rho", 1000.0)

    sub_obj2 = {
        "name": metadata["obj2"]["geometry"],
        "object_id": 2,
        "material": MATERIAL_MAP[obj2_material],
        **obj2_params,
        "rho": obj2_rho,  # Use rho from metadata
        "init_vel": [0.0, 0.0, 0.0],  # Set to zero as requested
        "color": metadata["obj2"]["surface_color"]
    }
    sub_objects.append(sub_obj2)

    # Combine trainable parameters
    all_trainable = list(set(obj1_trainable + obj2_trainable))
    params = {param: TRAINING_PARAMS[param] for param in all_trainable}

    # Build data section with bounds
    data_section = {
        "xyz_min": global_xyz_min,
        "xyz_max": global_xyz_max
    }

    # Add per-object bounds if available
    if obj1_xyz_min is not None:
        data_section["obj1_xyz_min"] = obj1_xyz_min
        data_section["obj1_xyz_max"] = obj1_xyz_max
        data_section["obj2_xyz_min"] = obj2_xyz_min
        data_section["obj2_xyz_max"] = obj2_xyz_max

    config = {
        "data": data_section,
        "gs": {
            "eval": True,
            "is_blender": True,
            "timenet": True,
            "test_iterations": [5000, 6000, 7000],
            "save_iterations": [7000, 10000, 20000, 30000, 40000],
            "quiet": False,
            "iterations": 40000,
            "enable_mask_training": True,
            "mask_loss_weight": 0.5
        },
        "physics": {
            "id": dataset_id,
            "fps": 24,  # Changed from 15 to 24 as requested
            "dt": 1.0 / (24 * 200),  # fps * mpm_iter_cnt
            "gravity": [0, -9.8, 0],
            "ground_friction": metadata.get("ground_friction", 0.5),

            "sub_objects": sub_objects,

            "voxel_size": voxel_size,
            "mpm_iter_cnt": 200,
            "bc": {
                "ground": [[0, 0, 0], [0, 1, 0], 0]
            },

            "density_grid_size": density_grid_size,
            "density_min_th": density_min_th,
            "density_max_th": density_max_th,
            "opacity_threshold": 0.01,
            "random_sample": False,

            "img_loss": True,
            "geo_loss": True,
            "w_img": 0.0,
            "w_alp": 1.0,
            "w_geo": 1.0,

            "params": params,
            "iter_cnt": 300,

            "vel_iter_cnt": 80,
            "vel_estimation_frames": 4,
            "vel_lr": 5.0e-2,
            "init_vel": [0.0, 0.0, 0.0],

            "collide_time": metadata.get("collide_time", 0.29166666666666663),
            "collide_loc": metadata.get("collide_loc", [0, 0, 0.25]),
            "n_frames": 20
        }
    }

    return config

def main():
    dataset_root = Path("/mnt/e/dataset_45_new_json")
    output_dir = Path("/mnt/c/Users/Chunjiang/OneDrive/Code/ICLR_2026/GIC_MO/config/genesismo")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all dataset folders
    dataset_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    created_configs = []
    skipped = []

    for dataset_folder in dataset_folders:
        dataset_id = dataset_folder.name

        # Skip 01 and 07 as they already exist
        if dataset_id in ["01", "07"]:
            skipped.append(dataset_id)
            continue

        metadata_path = dataset_folder / "metadata.json"
        if not metadata_path.exists():
            print(f"Warning: No metadata.json found for dataset {dataset_id}")
            continue

        try:
            config = create_config(dataset_id, metadata_path)

            # Save config
            output_path = output_dir / f"{dataset_id}.json"
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=4)

            created_configs.append(dataset_id)
            print(f"Created config for dataset {dataset_id}")

        except Exception as e:
            print(f"Error creating config for dataset {dataset_id}: {e}")

    print(f"\nSummary:")
    print(f"Created configs: {len(created_configs)} - {', '.join(created_configs)}")
    print(f"Skipped (already exist): {len(skipped)} - {', '.join(skipped)}")

if __name__ == "__main__":
    main()