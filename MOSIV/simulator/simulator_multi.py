"""
Simple wrapper around Estimator for multi-object prediction
No need to reimplement everything - just use Estimator without backward!
"""
from simulator.estimator_multi_beta import Estimator
import torch
import numpy as np


class MultiObjectSimulator:
    def __init__(self, estimation_params, vol, object_labels, config_file, device='cuda'):
        """
        Wrapper around Estimator for prediction only
        Args:
            estimation_params: Loaded prediction JSON with trained parameters
            vol: Particle positions
            object_labels: Object IDs for each particle
            config_file: Original training config file path (required)
            device: CUDA device
        """
        self.device = device
        self.vol = vol
        self.object_labels = object_labels

        # Load original training config to get all required fields
        import json
        from argparse import Namespace
        with open(config_file, 'r') as f:
            original_config = json.load(f)

        # Start with original config's phys section
        phys_args = Namespace(**original_config['physics'])

        # Update with trained values from estimation_params
        if hasattr(estimation_params, 'sub_objects'):
            # Replace sub_objects with the one from estimation_params (has trained values)
            phys_args.sub_objects = estimation_params.sub_objects

        # Copy other fields from estimation_params (might have been adjusted during training)
        for key in ['mpm_iter_cnt', 'voxel_size', 'gravity', 'bc', 'fps', 'density_grid_size']:
            if hasattr(estimation_params, key):
                setattr(phys_args, key, getattr(estimation_params, key))

        self.phys_args = phys_args

        # Fix the saved JSON format to match what Estimator expects
        if hasattr(phys_args, 'sub_objects'):
            for obj in phys_args.sub_objects:
                # Fix velocity naming: Estimator expects 'init_vel' but saved JSON has 'vel'
                if 'vel' in obj and 'init_vel' not in obj:
                    obj['init_vel'] = obj['vel']  # Copy trained velocity to init_vel

                # Fix material location: Estimator expects 'material' and 'rho' at top level
                if 'mat_params' in obj and 'material' not in obj:
                    obj['material'] = obj['mat_params']['material']
                    obj['rho'] = obj['mat_params'].get('rho', 1000.0)
                    # Also copy other material params to top level for Estimator
                    for key, value in obj['mat_params'].items():
                        if key not in obj:
                            obj[key] = value

        # Create dummy empty GT since we're only doing prediction
        dummy_gts = []

        # Create particle_materials from object_labels and sub_objects
        particle_materials = torch.zeros(len(vol), dtype=torch.int32, device=device)
        if hasattr(phys_args, 'sub_objects'):
            for obj in phys_args.sub_objects:
                obj_id = obj['object_id']
                # Get material type from mat_params (prediction) or direct (training)
                if 'mat_params' in obj:
                    material_type = obj['mat_params']['material']
                else:
                    material_type = obj.get('material', 10)
                # Assign material type to all particles of this object
                particle_materials[object_labels == obj_id] = material_type

        # Create Estimator (it already handles multi-object internally)
        self.estimator = Estimator(
            phys_args=phys_args,
            dtype='float32',
            gts=dummy_gts,  # Empty GT for prediction only
            init_vol=vol,
            gts_per_object=None,  # No GT needed
            surface_index=None,
            dynamic_scene=None,
            image_scale=1.0,
            pipeline=None,
            image_op=None,
            particle_materials=particle_materials,  # Per-particle materials
            object_labels=object_labels
        )

        # Store velocities for compatibility
        if hasattr(phys_args, 'sub_objects') and phys_args.sub_objects:
            first_obj = phys_args.sub_objects[0]
            self.vel = torch.tensor(first_obj.get('vel', first_obj.get('init_vel', [0,0,0])), device=device)
        else:
            self.vel = torch.zeros(3, device=device)

    def initialize(self, phys_args=None):
        """Initialize the simulator"""
        self.estimator.initialize()

    def forward(self, f):
        """Get particle positions at frame f"""
        # Call estimator forward without backward
        with torch.no_grad():
            particle_pos = self.estimator.forward(f, img_backward=False)

        # Convert to tensor
        if isinstance(particle_pos, np.ndarray):
            particle_pos = torch.from_numpy(particle_pos).to(self.device)

        return particle_pos

    def succeed(self):
        """Check if simulation succeeded"""
        return self.estimator.succeed()