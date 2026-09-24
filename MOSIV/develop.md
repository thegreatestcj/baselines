# Multi-Object Gaussian Splatting with Per-Instance Segmentation

## Overview
This document outlines the implementation plan for adding multi-object support with per-instance segmentation to the Gaussian Splatting framework. The key idea is to enable each Gaussian to learn object membership probabilities and render instance segmentation masks alongside RGB images.

## Architecture Changes

### 1. Per-Particle Material System (Completed)
- **Material Types**: 5 distinct materials (elasticity=10, viscous_fluid=11, von_mises=12, drucker_prager=13, non_newtonian=14)
- **Per-Particle Parameters**: All 8 physical properties are now per-particle
  - Elastic: E, nu → converted to mu, lam
  - Plastic: yield_stress, cohesion, friction_alpha, plastic_viscosity
- **Per-Particle Material Type**: Each particle has `p_material[p]` field

### 2. GenesisMO Multi-Object Dataset Support (Completed)

#### Dataset Format
The GenesisMO dataset introduces a multi-object format with the following structure:
```
/path/to/dataset/
├── metadata.json          # Scene configuration with object properties
├── all_data.json          # Camera parameters (same format as PAC-NeRF)
├── data/
│   ├── r_0_-1.png        # Images: r_{camera}_{frame}.png
│   └── ...
└── point_clouds/
    ├── 0/                 # Object 0 GT point clouds
    │   ├── 0.ply         # Frame 0
    │   └── ...
    └── 1/                 # Object 1 GT point clouds
        ├── 0.ply
        └── ...
```

#### Key Components in metadata.json
- `mpm_lower_bound`: Scene lower bounds (used as xyz_min)
- `mpm_upper_bound`: Scene upper bounds (used as xyz_max)
- `obj1`, `obj2`: Per-object physical properties (E, nu, rho, material type, colors)

#### Dataloader Implementation
Created `readGenesisMOInfo()` in `scene/dataset_readers.py`:
- **Inherits PAC-NeRF implementation**: Same logic for random point generation
- **Auto-detection**: Checks for `metadata.json` or "GenesisMO" in path
- **Config requirement**: Expects config with `xyz_min/xyz_max` from metadata bounds
- **Camera loading**: Uses `readCamerasFromAllData()` (same as PAC-NeRF)
- **Point cloud handling**:
  - Training (`load_fix_pcd=False`): Generates random points within bounds
  - Evaluation (`load_fix_pcd=True`): Returns None, loads trained Gaussians
  - GT point clouds in `point_clouds/` are loaded separately during physics training

#### Integration
Updated `scene/__init__.py` to automatically detect GenesisMO format:
```python
if os.path.exists(os.path.join(args.source_path, "metadata.json")) or "GenesisMO" in args.source_path:
    # Use GenesisMO reader
else:
    # Use PAC-NeRF reader
```

### 3. Instance Segmentation via Gaussian Attributes (Completed)

#### Core Idea
Each Gaussian learns three additional attributes representing object membership probabilities:
- Background (channel 0)
- Object 1 (channel 1)  
- Object 2 (channel 2)

These are stored as logits, converted to probabilities via softmax, and rendered as a 3-channel segmentation mask.

#### Implementation Details

##### 3.1 Extended Gaussian Model (`scene/gaussian_model.py`)

**Added Attributes:**
- `_object_logits`: `[N, 3]` tensor storing logits for [bg, obj1, obj2]
- Initialized to zeros (equal probabilities) in `create_from_pcd`
- Added to optimizer with feature learning rate

**New Properties:**
```python
@property
def get_object_logits(self):
    return self._object_logits

@property
def get_object_probs(self):
    """Convert object logits to probabilities via softmax"""
    return torch.softmax(self._object_logits, dim=-1)
```

**Initialization in `create_from_pcd`:**
```python
# Initialize object logits (bg, obj1, obj2)
object_logits = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
self._object_logits = nn.Parameter(object_logits.requires_grad_(True))
```

**Added to Optimizer in `training_setup`:**
```python
{'params': [self._object_logits], 'lr': training_args.feature_lr, "name": "object_logits"}
```

##### 3.2 Dual Rendering Pipeline (`gaussian_renderer/__init__.py`)

**Modified `render` function signature:**
```python
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, render_object_mask=False)
```

**Added dual rendering logic:**
```python
if render_object_mask:
    # Get object probabilities from Gaussian model
    object_probs = pc.get_object_probs  # [N, 3] softmax probabilities
    
    # Render mask using object probabilities as colors
    rendered_mask, _, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=object_probs,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
    
    results["mask"] = rendered_mask  # [3, H, W] with channels for [bg, obj1, obj2]
```

**Key Design Decisions:**
1. **Single function with flag**: Instead of separate functions, use `render_object_mask` parameter
2. **Same rasterizer**: Reuse the same rasterization pipeline for efficiency
3. **Override colors**: Use `colors_precomp` to pass object probabilities as "colors"
4. **Channel mapping**:
   - Channel 0: Background probability
   - Channel 1: Object 1 probability
   - Channel 2: Object 2 probability

##### 3.3 Usage Example
```python
# During training, enable dual rendering
results = render(
    viewpoint_camera=viewpoint,
    pc=gaussians,
    pipe=pipeline,
    bg_color=background,
    d_xyz=d_xyz,
    d_rotation=d_rotation,
    d_scaling=d_scaling,
    render_object_mask=True  # Enable mask rendering
)

# Access results
rgb_image = results["render"]  # [3, H, W] RGB image
mask_probs = results["mask"]   # [3, H, W] object probability maps
alpha = results["alpha"]        # [H, W] opacity/coverage (same for RGB and mask)
```

### 4. Object Mask Loading (Completed)

#### 4.1 Ground Truth Mask Loading Implementation

##### Modified CameraInfo Structure (`scene/dataset_readers.py`)
Added optional `object_mask` field to store `[3, H, W]` numpy array for segmentation masks.

##### Created `readCamerasFromAllDataWithMask` Function
New dataloader function specifically for GenesisMO that loads both RGB and mask images:

**Key Features:**
- **RGB images**: Loads with prefix `m_` (same as PAC-NeRF)
- **Object masks**: Loads with prefix `o_` 
- **File naming**: `o_{camera}_{frame}.png` for masks
- **Format handling**:
  - Grayscale masks: Converts to one-hot encoding (0=bg, 1=obj1, 2=obj2)
  - RGB masks: Direct use as probability channels [bg, obj1, obj2]
- **Resolution**: Resizes to 800x800 to match RGB images if needed
- **Data type**: Keeps as numpy arrays (no torch conversion in dataloader)

##### Updated GenesisMO Reader
- Now uses `readCamerasFromAllDataWithMask` for mask support
- Fully backward compatible with datasets without masks

### 5. Training Loop Integration (Completed)

#### 5.1 Created Multi-Object Training Function
Created `training_MO` function in `train_gs.py` with full mask support:

**Key Features:**
- Loads mask training hyperparameters from optimization args
- Conditionally renders masks based on iteration and mask availability
- Computes mask loss using same structure as RGB (L1 + SSIM)
- Combines RGB, alpha, and mask losses

**Implementation Details:**
```python
# Multi-object parameters from optimization args
enable_mask_training = getattr(opt, 'enable_mask_training', True)
mask_loss_weight = getattr(opt, 'mask_loss_weight', 0.5)
mask_start_iter = getattr(opt, 'mask_start_iter', 5000)

# Check if we should render masks
render_mask_flag = (enable_mask_training and 
                   hasattr(viewpoint_cam, 'object_mask') and 
                   viewpoint_cam.object_mask is not None and
                   iteration >= mask_start_iter)

# Render with optional mask
render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, 
                      d_xyz, d_rotation, d_scaling, dataset.is_6dof,
                      render_object_mask=render_mask_flag)

# Object mask loss
if render_mask_flag and "mask" in render_pkg_re:
    rendered_mask = render_pkg_re["mask"]
    gt_mask = torch.from_numpy(viewpoint_cam.object_mask).cuda().float()
    gt_mask = gt_mask.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
    
    # Use same loss structure as RGB: L1 + SSIM
    mask_l1 = l1_loss(rendered_mask, gt_mask)
    mask_ssim = ssim(rendered_mask, gt_mask)
    mask_loss = (1.0 - opt.lambda_dssim) * mask_l1 + opt.lambda_dssim * (1.0 - mask_ssim)
    loss += mask_loss_weight * mask_loss
```

#### 5.2 Updated Mask Loading for `.npy` Format
Modified `readCamerasFromAllDataWithMask` in `scene/dataset_readers.py`:

**Key Changes:**
- Loads 2-channel masks from `.npy` files (obj1, obj2)
- Adds zero background channel to create [H, W, 3] format
- Background channel is always 0.0 (no Gaussian represents background)

```python
# Load 2-channel mask from .npy file
mask_2ch = np.load(mask_path)  # Shape: [H, W, 2]

# Create 3-channel mask
mask_3ch = np.zeros((H, W, 3), dtype=np.float32)
mask_3ch[:, :, 0] = 0.0  # Background (always zero)
mask_3ch[:, :, 1] = mask_2ch[:, :, 0]  # Object 1
mask_3ch[:, :, 2] = mask_2ch[:, :, 1]  # Object 2
```

#### 5.3 Created train_dynamic_MO.py
- Imports `training_MO` instead of `training`
- Ready for multi-object Gaussian training with segmentation

### 6. Complete Multi-Object Pipeline Summary

The full pipeline is now operational with the following components:

1. **Data Loading**: Loads RGB and 2-channel object masks, adds zero background
2. **Gaussian Model**: Object logits with softmax conversion to probabilities
3. **Dual Rendering**: Simultaneous RGB and segmentation mask rendering
4. **Training Supervision**: Combined RGB + mask loss with configurable weights
5. **Background Handling**: Background rendered via background color, not Gaussians

### 7. Next Steps (Future Work)

#### 7.1 Visualization
- Save rendered masks during training for debugging
- Create comparison images: RGB | GT Mask | Predicted Mask
- Log mask accuracy metrics (mIoU)

#### 7.2 Save/Load PLY Support
- Extend PLY format to include object logits
- Handle object logits in `save_ply()` and `load_ply()` methods

#### 7.3 Hyperparameter Configuration
- Add mask training parameters to OptimizationParams
- Create config files with optimal settings

## Configuration Format
```json
{
    "physics": {
        "material": 12,  // Default material
        "sub_objects": [
            {
                "name": "elastic_cube",
                "object_id": 1,
                "material": 10,
                "xyz_min": [0.2, 0.4, 0.3],
                "xyz_max": [0.4, 0.6, 0.5],
                "init_E": 5.0,
                "init_nu": 0.3
            },
            {
                "name": "viscous_sphere",
                "object_id": 2,
                "material": 11,
                "xyz_min": [0.5, 0.4, 0.3],
                "xyz_max": [0.7, 0.6, 0.5],
                "mu": 1.0,
                "kappa": 1000.0
            }
        ]
    }
}
```

## Benefits of This Approach

1. **Differentiable**: Softmax probabilities allow gradient flow for learning object boundaries
2. **Soft Boundaries**: Natural handling of ambiguous regions between objects
3. **Unified Rendering**: Uses existing render function with override_color
4. **No CUDA Modifications**: Works with current rasterizer
5. **Extensible**: Easy to add more objects (just add more logits)

## Training Strategy

1. **Initialization**: Use spatial regions to initialize object logits
2. **Warm-up**: Train RGB first, then add segmentation loss
3. **Loss Weighting**: Start with small weight for segmentation loss, increase gradually
4. **Regularization**: Add entropy regularization to encourage decisive predictions:
   ```python
   entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
   loss_reg = -0.01 * entropy  # Negative to minimize entropy
   ```

## Evaluation Metrics

- **mIoU**: Mean Intersection over Union for segmentation quality
- **Boundary Accuracy**: Accuracy at object boundaries
- **Instance Separation**: How well objects are separated

## Future Extensions

1. **Dynamic Object Count**: Learn number of objects automatically
2. **Hierarchical Segmentation**: Objects → Parts → Materials
3. **Temporal Consistency**: Enforce consistent IDs across frames
4. **Interactive Editing**: Select and manipulate individual objects

## Per-Object Geometry Loss Implementation

### Approach: Call existing functions multiple times with filtered inputs

Instead of modifying the core loss calculation functions (`compute_loss_gt2sim`, `compute_loss_sim2gt`), we can achieve per-object loss by:
1. Filtering particles by object ID
2. Calling existing functions for each object separately
3. Accumulating per-object losses

### Complete Implementation for Per-Object Geometry Loss

```python
def forward_with_per_object_loss(self, f, img_backward=True):
    """
    Modified forward pass that computes geometry loss separately for each object.
    This implementation reuses existing Taichi kernels without modification.
    """
    particle_pos = np.zeros([self.num_particles[None], 3], dtype=np.float32)
    
    if f > 0:
        self.simulator.advance(f - 1)
    
    if not self.succeed():
        return particle_pos
    
    local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
    
    # Per-object geometry loss computation
    if len(self.gts) > 0 and (self.geo_loss or self.stage[None] == self.velocity_stage):
        # Assuming we have per-object GTs from prepare_gt_multi
        # self.gts_per_object = {1: gts_obj1, 2: gts_obj2}
        # Each contains list of frames with GT surface points
        
        total_loss = 0.0
        per_object_losses = {}
        
        # Process each object separately
        for obj_id in [1, 2]:  # For 2 objects, extend as needed
            
            # Step 1: Filter simulation surface particles for this object
            obj_sim_indices = []
            for i in range(self.sim_surface_cnt[f]):
                particle_idx = self.sim_surface_index[f, i]
                # Check if this particle belongs to current object
                if self.object_labels[particle_idx] == obj_id:
                    obj_sim_indices.append(particle_idx)
            
            # Step 2: Get GT particles for this object
            # Assuming gts_per_object structure from prepare_gt_multi
            if hasattr(self, 'gts_per_object') and obj_id in self.gts_per_object:
                if f < len(self.gts_per_object[obj_id]):
                    gt_points_for_obj = self.gts_per_object[obj_id][f]
                else:
                    gt_points_for_obj = []
            else:
                # Fallback: use object labels if available
                gt_points_for_obj = []
                if hasattr(self, 'gt_object_labels'):
                    for i in range(self.num_particles_surface[f]):
                        if self.gt_object_labels[f, i] == obj_id:
                            gt_points_for_obj.append(self.gt[f, i])
            
            # Skip if no particles for this object
            if len(obj_sim_indices) == 0 or len(gt_points_for_obj) == 0:
                per_object_losses[obj_id] = 0.0
                continue
            
            # Step 3: Backup current arrays and error counts
            # Need to use to_numpy() for Taichi fields
            backup_sim_surface = np.zeros_like(self.sim_surface_index.to_numpy()[f])
            backup_sim_surface[:] = self.sim_surface_index.to_numpy()[f]
            backup_sim_cnt = self.sim_surface_cnt[f]
            
            backup_gt = np.zeros((self.num_particles_surface[f], 3), dtype=np.float32)
            for i in range(self.num_particles_surface[f]):
                for d in range(3):
                    backup_gt[i, d] = self.gt[f, i][d]
            backup_gt_cnt = self.num_particles_surface[f]
            
            # Backup error counts
            backup_gt2sim_err_cnt = self.gt2sim_err_cnt[f]
            backup_sim2gt_err_cnt = self.sim2gt_err_cnt[f]
            
            # Step 4: Fill arrays with filtered data for this object
            # Update sim surface indices
            for i, idx in enumerate(obj_sim_indices):
                self.sim_surface_index[f, i] = idx
            self.sim_surface_cnt[f] = len(obj_sim_indices)
            
            # Update GT points
            for i, point in enumerate(gt_points_for_obj):
                for d in range(3):
                    self.gt[f, i][d] = point[d] if hasattr(point, '__getitem__') else point.numpy()[d]
            self.num_particles_surface[f] = len(gt_points_for_obj)
            
            # Step 5: Reset loss accumulator and compute loss for this object
            self.loss[None] = 0.0
            
            # Update matching indices
            self.update_match_indices_gt2sim(f, local_index)
            self.update_match_indices_sim2gt(f, local_index)
            
            # Compute bidirectional losses
            self.compute_loss_gt2sim(f, local_index)
            self.compute_loss_sim2gt(f, local_index)
            
            # Store this object's loss
            per_object_losses[obj_id] = self.loss[None]
            
            # Optional: Store error counts for debugging
            if not hasattr(self, 'per_object_err_counts'):
                self.per_object_err_counts = {}
            self.per_object_err_counts[obj_id] = {
                'gt2sim_err_cnt': self.gt2sim_err_cnt[f],
                'sim2gt_err_cnt': self.sim2gt_err_cnt[f]
            }
            
            # Step 6: Restore original arrays
            for i in range(backup_sim_cnt):
                self.sim_surface_index[f, i] = backup_sim_surface[i]
            self.sim_surface_cnt[f] = backup_sim_cnt
            
            for i in range(backup_gt_cnt):
                for d in range(3):
                    self.gt[f, i][d] = backup_gt[i, d]
            self.num_particles_surface[f] = backup_gt_cnt
            
            # Restore error counts
            self.gt2sim_err_cnt[f] = backup_gt2sim_err_cnt
            self.sim2gt_err_cnt[f] = backup_sim2gt_err_cnt
        
        # Combine per-object losses
        self.loss[None] = sum(per_object_losses.values())
        
        # Store for logging/debugging
        self.per_object_geo_losses = per_object_losses
        
        # Optional: Print per-object losses for debugging
        if f % 10 == 0:  # Print every 10 frames
            loss_str = ", ".join([f"Obj{k}: {v:.4f}" for k, v in per_object_losses.items()])
            print(f"Frame {f} - Per-object geo losses: {loss_str}")
    
    # Get particle positions and continue with rendering
    self.simulator.get_x(f, particle_pos)
    particle_pos = torch.from_numpy(particle_pos).to(self.device).requires_grad_()
    
    # Render if needed
    if (f > 0 and self.stage[None] == self.physical_params_stage and self.img_loss) or \
            (f > 0 and self.stage[None] == self.velocity_stage and len(self.gts) == 0):
        self.render_forward(f, particle_pos, img_backward)
    
    return particle_pos


def backward_with_per_object(self, f):
    """
    Modified backward pass that handles per-object gradients.
    """
    local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
    
    if self.stage[None] == self.physical_params_stage and f > 0 and self.img_loss or (len(self.gts) == 0):
        self.set_pos_grad(f, self.pos_grad_seq[f - 1].data.cpu().numpy())
    
    # Per-object gradient computation
    if self.geo_loss or self.stage[None] == self.velocity_stage:
        # Process each object separately for gradients
        for obj_id in [1, 2]:
            # Similar filtering and backup/restore as in forward pass
            # But for gradient computation
            
            # Filter sim surface particles for this object
            obj_sim_indices = []
            for i in range(self.sim_surface_cnt[f]):
                particle_idx = self.sim_surface_index[f, i]
                if self.object_labels[particle_idx] == obj_id:
                    obj_sim_indices.append(particle_idx)
            
            # Get GT particles for this object
            if hasattr(self, 'gts_per_object') and obj_id in self.gts_per_object:
                if f < len(self.gts_per_object[obj_id]):
                    gt_points_for_obj = self.gts_per_object[obj_id][f]
                else:
                    continue
            else:
                continue
            
            if len(obj_sim_indices) == 0 or len(gt_points_for_obj) == 0:
                continue
            
            # Backup and replace arrays (similar to forward)
            # ... (same backup/restore logic as forward)
            
            # Compute gradients for this object
            self.compute_loss_sim2gt.grad(f, local_index)
            self.compute_loss_gt2sim.grad(f, local_index)
            
            # Restore arrays
            # ... (same restore logic as forward)
    
    if f > 0:
        self.simulator.advance_grad(f - 1)
    else:
        # Handle initial frame gradients
        self.compute_particle_mass.grad()
        # ... rest of gradient handling
```

### Key Implementation Details

1. **Array Backup/Restore**:
   - Use `to_numpy()` for Taichi fields to create proper backups
   - Manually copy each element when restoring (Taichi fields don't support slice assignment)

2. **Error Counts**:
   - `gt2sim_err_cnt` and `sim2gt_err_cnt` track particles above ground
   - These need to be backed up and restored too, as they're used for loss normalization

3. **GT Data Structure**:
   - Option A: `self.gts_per_object = {1: [...], 2: [...]}` from `prepare_gt_multi`
   - Option B: `self.gt_object_labels[f, i]` tracking which GT particle belongs to which object

4. **Loss Combination**:
   - Simple sum: `total_loss = sum(per_object_losses.values())`
   - Weighted: `total_loss = sum(loss * weight for loss, weight in zip(losses, weights))`
   - Normalized: `total_loss = sum(loss / particle_count for loss, count in ...)`

5. **Gradient Flow**:
   - The gradients are accumulated correctly because we're using the same Taichi autodiff
   - Each object's gradients are computed separately but accumulated in the same fields

6. **Performance Considerations**:
   - Array copying is fast (GPU memory operations)
   - Main overhead is running kernels twice (once per object)
   - Can optimize by parallelizing if kernels support concurrent execution

### Requirements for Implementation

1. **GT Object Labels**: Need to track which GT particles belong to which object
   - Option A: Separate GT loading for each object (already done in `prepare_gt_multi`)
   - Option B: Add object ID field for GT particles

2. **Filtering Functions**: Need helper functions to filter particles by object
   ```python
   @ti.kernel
   def filter_surface_by_object(surface_index, object_labels, target_obj_id):
       # Return indices of particles belonging to target_obj_id
       pass
   ```

3. **Loss Weighting**: Decide how to combine per-object losses
   - Equal weight: `total_loss = sum(losses) / num_objects`
   - Weighted by particle count: `total_loss = weighted_sum(losses, particle_counts)`
   - Custom weights: `total_loss = sum(loss * weight for loss, weight in zip(losses, object_weights))`

### Alternative Approach: Modify Loss Functions with Object ID

```python
@ti.kernel
def compute_loss_gt2sim_per_object(self, f, local_index, obj_id):
    """
    Compute loss only for particles belonging to obj_id
    """
    for i in range(self.num_particles_surface[f]):
        # Check if GT particle belongs to this object
        if self.gt_object_ids[f, i] != obj_id:
            continue
            
        index_ = self.match_indices_gt2sim[f, i]
        
        # Check if matched sim particle belongs to this object
        if self.object_labels[index_] != obj_id:
            continue
            
        d = self.compute_distance(self.simulator.x[index_, local_index], self.gt[f, i])
        
        # Accumulate to per-object loss field
        self.per_object_loss[obj_id] += d * self.w_geo[None] / self.per_object_gt2sim_cnt[f, obj_id]
```

### Pros and Cons

**Approach 1: Multiple Calls with Filtering**
- Pros: 
  - No modification to core functions
  - Clean separation of concerns
  - Easy to toggle on/off
- Cons:
  - Multiple passes over data (slower)
  - Need temporary data swapping
  - More complex state management

**Approach 2: Modified Functions with Object ID**
- Pros:
  - Single pass over data (faster)
  - Direct per-object loss computation
  - Cleaner code flow
- Cons:
  - Need to modify core functions
  - More invasive changes
  - Harder to maintain compatibility

### Recommendation

Start with Approach 1 (multiple calls) for proof of concept, then optimize to Approach 2 if performance becomes an issue.

### Testing Strategy

1. Verify that sum of per-object losses equals original combined loss
2. Test with objects of different sizes to ensure proper normalization
3. Visualize per-object matching to ensure correct particle assignment
4. Compare convergence with and without per-object loss