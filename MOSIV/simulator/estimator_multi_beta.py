# coding=utf-8

import torch
import torchvision
import numpy as np
import taichi as ti
import torch.nn as nn
from simulator import MPMSimulator
from gaussian_renderer import render
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
import os


def constraint(x, bound):
    # return x
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0] + y_scale)


def constraint_inv(y, bound):
    # return y
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale


@ti.data_oriented
class Estimator(torch.nn.Module):
    velocity_stage = 0
    physical_params_stage = 1

    def __init__(self, phys_args, dtype, gts: list, init_vol,
                 surface_index=None, cuda_chunk_size=100, dynamic_scene=None,
                 image_scale=1.0, pipeline=None, image_op=None,
                 particle_materials=None, object_labels=None, gts_per_object=None):
        super(Estimator, self).__init__()
        self.pipeline = pipeline
        self.image_op = image_op
        self.image_scale = image_scale
        self.stage = ti.field(ti.int32, shape=())
        self.stage[None] = -1
        self.args = phys_args
        self.particle_materials = particle_materials  # Use per-particle materials directly
        self.voxel_size = phys_args.voxel_size
        self.dtype = ti.f64 if dtype == 'float64' else ti.f32
        frame_dt = self.frame_dt = 1.0 / phys_args.fps
        dt = frame_dt / phys_args.mpm_iter_cnt
        gravity = phys_args.gravity
        self.img_loss = getattr(phys_args, 'img_loss', True)
        self.geo_loss = getattr(phys_args, 'geo_loss', True)
        self.w_img = torch.tensor(getattr(phys_args, "w_img", 0.0), device=init_vol.device)
        self.w_alp = torch.tensor(getattr(phys_args, "w_alp", 0.0), device=init_vol.device)
        self.w_geo = ti.field(ti.f32, shape=(), needs_grad=True)
        self.w_geo[None] = getattr(phys_args, "w_geo", 1.0)

        self.num_particles = ti.field(ti.i32, shape=())

        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.frame_dt = frame_dt
        self.gts = gts
        self.gts_per_object = gts_per_object
        self.max_f = len(gts)
        max_f = self.max_f if self.max_f > 0 else 1
        particles_count, _ = init_vol.shape
        print(f'obj particles count: {particles_count}')

        surface_count = self.get_gts_surface_count(gts)
        surface_count = surface_count if surface_count > 0 else 1
        self.num_particles[None] = particles_count
        self.device = init_vol.device
        self.init_vol = init_vol

        # max mpm surface particles count
        self.sim_surface_particles_cnt = sim_surface_particles_cnt = self.get_surface_particles_cnt(
            surface_index)  # surface_index.shape[0]

        self.sim_surface_index = ti.field(dtype=ti.i32, shape=(max_f, sim_surface_particles_cnt))
        self.sim_surface_cnt = ti.field(dtype=ti.i32, shape=max_f)  # mpm surface particles count per frame

        self.gt = ti.Vector.field(n=3, dtype=self.dtype, shape=(max_f, surface_count), needs_grad=True)

        # Combined fields (kept for compatibility)
        self.match_indices_gt2sim = ti.field(ti.i32, shape=(max_f, surface_count))
        self.match_indices_sim2gt = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))
        self.gt2sim_err_cnt = ti.field(ti.i32, shape=max_f)
        self.sim2gt_err_cnt = ti.field(ti.i32, shape=max_f)

        # Per-object match indices and error counts
        self.match_indices_gt2sim_obj1 = ti.field(ti.i32, shape=(max_f, surface_count))
        self.match_indices_sim2gt_obj1 = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))
        self.gt2sim_err_cnt_obj1 = ti.field(ti.i32, shape=max_f)
        self.sim2gt_err_cnt_obj1 = ti.field(ti.i32, shape=max_f)

        self.match_indices_gt2sim_obj2 = ti.field(ti.i32, shape=(max_f, surface_count))
        self.match_indices_sim2gt_obj2 = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))
        self.gt2sim_err_cnt_obj2 = ti.field(ti.i32, shape=max_f)
        self.sim2gt_err_cnt_obj2 = ti.field(ti.i32, shape=max_f)

        self.num_particles_surface = ti.field(ti.i32, shape=(max_f))  # gt surface particles count per frame
        self.num_particles[None] = particles_count
        # self.num_particles_surface[None] = surface_count

        # Per-object surface indices
        self.sim_surface_index_obj1 = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))
        self.sim_surface_index_obj2 = ti.field(ti.i32, shape=(max_f, self.sim_surface_particles_cnt))
        self.sim_surface_cnt_obj1 = ti.field(ti.i32, shape=(max_f))
        self.sim_surface_cnt_obj2 = ti.field(ti.i32, shape=(max_f))

        # Store object labels early (needed for surface indices creation)
        self.object_labels = object_labels if object_labels is not None else None

        # Per-object GT data
        self.gts_per_object = gts_per_object if gts_per_object is not None else None
        if self.gts_per_object:
            # Object 1 GT
            max_gt_obj1 = max(len(frame_gts) for frame_gts in self.gts_per_object[1])
            self.gt_obj1 = ti.Vector.field(n=3, dtype=self.dtype, shape=(max_f, max_gt_obj1), needs_grad=True)
            self.num_particles_surface_obj1 = ti.field(ti.i32, shape=(max_f))

            # Object 2 GT
            max_gt_obj2 = max(len(frame_gts) for frame_gts in self.gts_per_object[2])
            self.gt_obj2 = ti.Vector.field(n=3, dtype=self.dtype, shape=(max_f, max_gt_obj2), needs_grad=True)
            self.num_particles_surface_obj2 = ti.field(ti.i32, shape=(max_f))

        self.load_gts(gts)
        if self.gts_per_object:
            self.load_gts_per_object()
        self.load_all_sim_surfaces(surface_index, max_f)
        if self.object_labels is not None:
            self.create_per_object_surface_indices(surface_index, max_f)

        self.particle_rho = ti.field(dtype=self.dtype, needs_grad=True)
        self.loss = ti.field(self.dtype, shape=(), needs_grad=True)

        # Per-object geometry losses
        self.loss_obj1 = ti.field(self.dtype, shape=(), needs_grad=True)
        self.loss_obj2 = ti.field(self.dtype, shape=(), needs_grad=True)

        self.grid_particles_density = ti.field(self.dtype, needs_grad=True)
        grid_size = 4096
        offset = tuple(-grid_size // 2 for _ in range(3))
        grid_block_size = 128
        leaf_block_size = 4

        grid = self.grid_observe = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)
        block.dense(ti.ijk, leaf_block_size).place(self.grid_particles_density, self.grid_particles_density.grad,
                                                   offset=offset)

        particle_chunk_size = 2 ** 14
        self.particle = ti.root.dynamic(ti.i, 2 ** 30, particle_chunk_size)
        self.particle.place(self.particle_rho, self.particle_rho.grad)

        self.init_rhos = None
        self.init_yield_stress = None
        self.init_plastic_viscosity = None
        self.init_friction_alpha = None
        self.init_cohesion = None
        self.nu_bound = getattr(phys_args, "nu_bound", [-0.99, 0.5])
        self.phys_args = phys_args  # Store phys_args for later use

        # Create trainable parameter dict per object
        self.training_params = nn.ModuleDict()
        particle_materials_np = particle_materials.cpu().numpy() if torch.is_tensor(
            particle_materials) else particle_materials
        object_labels_np = object_labels.cpu().numpy() if torch.is_tensor(object_labels) else object_labels

        # Create per-object trainable parameters based on sub_objects config
        for obj_config in phys_args.sub_objects:
            obj_id = obj_config['object_id']
            obj_material = obj_config['material']
            obj_mask = object_labels_np == obj_id
            n_obj_particles = obj_mask.sum()

            if n_obj_particles > 0:
                obj_params = nn.ParameterDict()
                obj_params['material'] = obj_material  # Store material type
                obj_params['n_particles'] = n_obj_particles  # Store particle count for this object

                # Use UNIFIED parameters per object (single value, not per-particle arrays)
                if obj_material == MPMSimulator.elasticity:  # Elastic: E, nu
                    # Store log(E) for optimization, not E directly
                    init_E = obj_config.get('init_E', 50000.0)
                    obj_params['E'] = nn.Parameter(torch.tensor(np.log10(init_E), device=self.device))
                    obj_params['nu'] = nn.Parameter(
                        constraint_inv(torch.tensor(obj_config.get('init_nu', 0.25), device=self.device),
                                       self.nu_bound))

                elif obj_material == MPMSimulator.von_mises:  # Von Mises (plasticine): E, nu, yield_stress
                    # Store log values for optimization
                    init_E = obj_config.get('init_E', 10000.0)
                    init_yield = obj_config.get('init_yield_stress', 1000.0)
                    obj_params['E'] = nn.Parameter(torch.tensor(np.log10(init_E), device=self.device))
                    obj_params['nu'] = nn.Parameter(
                        constraint_inv(torch.tensor(obj_config.get('init_nu', 0.25), device=self.device),
                                       self.nu_bound))
                    obj_params['yield_stress'] = nn.Parameter(torch.tensor(np.log10(init_yield), device=self.device))

                elif obj_material == MPMSimulator.drucker_prager:  # Drucker-Prager (sand): E, nu, friction_alpha
                    # Store log(E) for optimization
                    init_E = obj_config.get('init_E', 10000.0)
                    obj_params['E'] = nn.Parameter(torch.tensor(np.log10(init_E), device=self.device))
                    obj_params['nu'] = nn.Parameter(
                        constraint_inv(torch.tensor(obj_config.get('init_nu', 0.25), device=self.device),
                                       self.nu_bound))
                    obj_params['friction_alpha'] = nn.Parameter(
                        torch.tensor(obj_config.get('init_friction_alpha', 30.0), device=self.device))
                    obj_params['cohesion'] = torch.tensor(0.0,
                                                          device=self.device)  # Cohesion is not trainable, always zero

                elif obj_material == MPMSimulator.viscous_fluid:  # Viscous fluid (Newtonian): kappa, mu
                    obj_params['mu'] = nn.Parameter(
                        torch.tensor(np.log10(obj_config.get('mu', 10.0)), device=self.device))
                    obj_params['kappa'] = nn.Parameter(
                        torch.tensor(np.log10(obj_config.get('kappa', 10000.0)), device=self.device))

                elif obj_material == MPMSimulator.non_newtonian:  # Non-Newtonian: kappa, mu, plastic_viscosity, yield_stress
                    # Store log values for optimization
                    init_viscosity = obj_config.get('init_plastic_viscosity', 100.0)
                    init_yield = obj_config.get('init_yield_stress', 1000.0)
                    obj_params['mu'] = nn.Parameter(
                        torch.tensor(np.log10(obj_config.get('mu', 10.0)), device=self.device))
                    obj_params['kappa'] = nn.Parameter(
                        torch.tensor(np.log10(obj_config.get('kappa', 10000.0)), device=self.device))
                    obj_params['plastic_viscosity'] = nn.Parameter(
                        torch.tensor(np.log10(init_viscosity), device=self.device))
                    obj_params['yield_stress'] = nn.Parameter(torch.tensor(np.log10(init_yield), device=self.device))

                self.training_params[str(obj_id)] = obj_params

        self.object_velocities = nn.ParameterDict()
        for obj_config in phys_args.sub_objects:
            obj_id = str(obj_config['object_id'])
            init_vel = obj_config.get('init_vel', [0.0, 0.0, 0.0])  # Use config velocity
            self.object_velocities[obj_id] = nn.Parameter(torch.tensor(init_vel, device=self.device))

        self.init_omega = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=self.device))
        # Use first object's material as default for simulator initialization
        default_material = phys_args.sub_objects[0]['material'] if hasattr(phys_args, 'sub_objects') else 10
        self.simulator = MPMSimulator(dtype=self.dtype, dt=dt, frame_dt=frame_dt, n_particles=self.num_particles,
                                      material=default_material, dx=self.dx, inv_dx=self.inv_dx,
                                      particle_layout=self.particle, args=phys_args, gravity=gravity,
                                      cuda_chunk_size=cuda_chunk_size)

        for collider_type, collider in phys_args.bc.items():
            if "ground" in collider_type:
                point, normal, bc_style = collider
                self.simulator.add_surface_collider(point, normal, bc_style)
            elif "cylinder" in collider_type:
                start, end, radius, bc_style = collider
                self.simulator.add_cylinder_collider(start, end, radius, bc_style)

        params = []
        # Add all trainable parameters from each object
        for obj_id_str, obj_params in self.training_params.items():
            for param_name, param_value in obj_params.items():
                if param_name == 'material':  # Skip non-parameter material type
                    continue
                # Map internal param names to optimizer param names
                if param_name == 'E':
                    opt_name = f"obj{obj_id_str}_Youngs_modulus"
                    lr = phys_args.params.get("Youngs modulus", {}).get('init_lr', 0.1)
                elif param_name == 'nu':
                    opt_name = f"obj{obj_id_str}_Poisson_ratio"
                    lr = phys_args.params.get("Poisson ratio", {}).get('init_lr', 0.025)
                elif param_name == 'kappa':
                    opt_name = f"obj{obj_id_str}_bulk_modulus"
                    lr = phys_args.params.get("bulk modulus", {}).get('init_lr', 0.1)
                elif param_name == 'mu':
                    opt_name = f"obj{obj_id_str}_shear_modulus"
                    lr = phys_args.params.get("shear modulus", {}).get('init_lr', 0.1)
                elif param_name == 'yield_stress':
                    opt_name = f"obj{obj_id_str}_Yield_stress"
                    lr = phys_args.params.get("Yield stress", {}).get('init_lr', 0.1)
                elif param_name == 'plastic_viscosity':
                    opt_name = f"obj{obj_id_str}_plastic_viscosity"
                    lr = phys_args.params.get("plastic viscosity", {}).get('init_lr', 0.05)
                elif param_name == 'friction_alpha':
                    opt_name = f"obj{obj_id_str}_friction_angle"
                    lr = phys_args.params.get("friction angle", {}).get('init_lr', 1.0)
                elif param_name == 'cohesion':
                    continue  # Cohesion is not trainable
                else:
                    continue
                if param_name != 'cohesion':  # Only add trainable parameters
                    params.append({'params': param_value, 'lr': lr, 'name': opt_name})
        # params.append({'params':self.init_omega, 'lr':info.get('init_lr', 1.5), 'name': "omega"})
        # Use lower momentum for better control with constraints
        self.optimizer = torch.optim.Adam([*params], betas=(0.7, 0.999), amsgrad=False)
        # Create velocity optimizer with per-object velocities
        vel_params = []
        self.vel_initial_lrs = {}  # Store initial learning rates for velocity
        for obj_id, vel_param in self.object_velocities.items():
            vel_params.append({'params': vel_param, 'lr': phys_args.vel_lr, 'name': f'velocity_obj_{obj_id}'})
            self.vel_initial_lrs[f'velocity_obj_{obj_id}'] = phys_args.vel_lr
        self.vel_optimizer = torch.optim.Adam(vel_params)
        self.lr_schedulers = {}
        for param_name, info in phys_args.params.items():
            if info.get('lr_decay', False):
                lr_init = info.get('init_lr', 0.1)
                lr_final = info.get('final_lr', 0.01)
                max_steps = info.get('max_steps', 60)
                self.lr_schedulers[param_name] = get_expon_lr_func(lr_init=lr_init, lr_final=lr_final,
                                                                   max_steps=max_steps, lr_delay_mult=1.0)
        self.config_id = phys_args.id
        self.pos_grad_seq = []
        self.image_loss = 0.0
        self.per_object_sil_losses_accum = {1: 0.0, 2: 0.0}  # Accumulated per-object losses
        self.views = None
        self.scene = dynamic_scene

    def scene():
        def fget(self):
            return self._scene

        def fset(self, value):
            if value is None:
                return
            self._scene = value
            views = self.scene.getTrainCameras(scale=self.image_scale)
            t_ls = torch.unique(torch.stack([view.fid for view in views if view.fid >= 0]))
            t_ls, _ = torch.sort(t_ls.cpu())
            all_views = []
            for t in t_ls:
                views_by_t = [v for v in views if torch.abs(v.fid.cpu() - t) < 1e-7]
                all_views.append(views_by_t)

            self.views = all_views

        return locals()

    scene = property(**scene())

    def zero_grad(self):
        if self.stage[None] == self.velocity_stage:
            self.vel_optimizer.zero_grad()
        elif self.stage[None] == self.physical_params_stage:
            self.optimizer.zero_grad()

    def step(self, i):
        if self.stage[None] == self.velocity_stage:
            self.vel_optimizer.step()
            if i >= 30 and i <= 80:
                decay_factor = 1.0 - (i - 30) / 50.0 * 0.9  # Goes from 1.0 to 0.1
                for param_group in self.vel_optimizer.param_groups:
                    original_lr = self.vel_initial_lrs.get(param_group['name'], self.phys_args.vel_lr)
                    param_group['lr'] = original_lr * decay_factor
        elif self.stage[None] == self.physical_params_stage:
            self.optimizer.step()
            self.update_learning_rate(i)

    def get_optimizer(self):
        if self.stage[None] == self.velocity_stage:
            return self.vel_optimizer
        elif self.stage[None] == self.physical_params_stage:
            return self.optimizer

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # Extract base parameter name from optimizer name (e.g., "obj1_Youngs_modulus" -> "Youngs modulus")
            name_parts = param_group["name"].split('_', 1)
            if len(name_parts) >= 2:
                base_name = name_parts[1].replace('_', ' ')  # "Youngs_modulus" -> "Youngs modulus"
                if base_name in self.lr_schedulers:
                    f = self.lr_schedulers[base_name]
                    lr = f(iteration)
                    param_group['lr'] = lr

    def set_scene(self, scene):
        self.scene = scene

    def set_stage(self, stage):
        self.stage[None] = stage

    @ti.kernel
    def get_input_grad(self, position_grad: ti.types.ndarray(),
                       velocity_grad: ti.types.ndarray(), rho_grad: ti.types.ndarray(),
                       mu_grad: ti.types.ndarray(), lam_grad: ti.types.ndarray(),
                       yield_stress_grad: ti.types.ndarray(),
                       viscosity_grad: ti.types.ndarray(),
                       friction_alpha_grad: ti.types.ndarray(),
                       cohesion_grad: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            rho_grad[p] = self.particle_rho.grad[p]
            mu_grad[p] = self.simulator.mu.grad[p]
            lam_grad[p] = self.simulator.lam.grad[p]
            yield_stress_grad[p] = self.simulator.yield_stress.grad[p]
            viscosity_grad[p] = self.simulator.plastic_viscosity.grad[p]
            friction_alpha_grad[p] = self.simulator.friction_alpha.grad[p]
            cohesion_grad[p] = self.simulator.cohesion.grad[p]
            for d in ti.static(range(3)):
                velocity_grad[p, d] = self.simulator.v.grad[p, 0][d]
                position_grad[p, d] = self.simulator.x.grad[p, 0][d]

    def get_nu(self):
        # Return constrained nu values for all objects
        nu_values = {}
        for obj_id_str, obj_params in self.training_params.items():
            if 'nu' in obj_params:
                nu_values[obj_id_str] = constraint(obj_params['nu'], self.nu_bound)
        return nu_values

    def get_surface_particles_cnt(self, surfaces):
        if surfaces is None:
            return self.num_particles[None]
        elif type(surfaces) == type([]):
            cnt = 0
            for idx in range(len(surfaces)):
                surface_p_cnt, _ = surfaces[idx].shape
                if surface_p_cnt > cnt:
                    cnt = surface_p_cnt
            return cnt
        else:
            return surfaces.shape[0]

    @staticmethod
    def get_gts_surface_count(gts):
        count = 0
        for idx in range(len(gts)):
            surface_count, _ = gts[idx].shape
            if surface_count > count:
                count = surface_count
        return count

    def load_gts(self, gts):
        for idx in range(len(gts)):
            surface_count, _ = gts[idx].shape
            self.num_particles_surface[idx] = surface_count
            self.save_gt(idx, gts[idx], surface_count)

    def load_gts_per_object(self):
        """Load per-object GT data into separate fields"""
        if self.gts_per_object is None:
            return

        # Load GT for object 1
        for idx in range(len(self.gts_per_object[1])):
            gt_obj1 = self.gts_per_object[1][idx]
            surface_count, _ = gt_obj1.shape
            self.num_particles_surface_obj1[idx] = surface_count
            self.save_gt_obj(idx, gt_obj1, surface_count, obj_id=1)

        # Load GT for object 2
        for idx in range(len(self.gts_per_object[2])):
            gt_obj2 = self.gts_per_object[2][idx]
            surface_count, _ = gt_obj2.shape
            self.num_particles_surface_obj2[idx] = surface_count
            self.save_gt_obj(idx, gt_obj2, surface_count, obj_id=2)

    @ti.kernel
    def save_gt_obj(self, f: ti.int32, x: ti.types.ndarray(), cnt: ti.int32, obj_id: ti.int32):
        for i in range(cnt):
            for j in ti.static(range(3)):
                if obj_id == 1:
                    self.gt_obj1[f, i][j] = x[i, j]
                else:  # obj_id == 2
                    self.gt_obj2[f, i][j] = x[i, j]

    def load_all_sim_surfaces(self, surfaces, max_f):
        if surfaces is None:
            surfaces = torch.arange(self.num_particles[None])
            # self.num_particles[None]

        surfaces = surfaces.data.cpu().numpy()
        cnt = max_f
        if type(surfaces) == type([]):
            cnt = len(surfaces)
        for idx in range(cnt):
            surface = surfaces[idx] if type(surfaces) == type([]) else surfaces
            self.sim_surface_cnt[idx] = surface.shape[0]
            self.load_sim_surface(surface, surface.shape[0], idx)

    @ti.kernel
    def load_sim_surface(self, surface_index: ti.types.ndarray(), cnt: ti.int32, f: ti.int32):
        for i in range(cnt):
            self.sim_surface_index[f, i] = surface_index[i]

    def create_per_object_surface_indices(self, surfaces, max_f):
        """Create per-object surface indices based on object labels"""
        if self.object_labels is None:
            return

        object_labels_np = self.object_labels.cpu().numpy() if torch.is_tensor(self.object_labels) else self.object_labels

        if surfaces is None:
            surfaces = torch.arange(self.num_particles[None])

        surfaces = surfaces.data.cpu().numpy() if hasattr(surfaces, 'data') else surfaces
        cnt = max_f
        if type(surfaces) == type([]):
            cnt = len(surfaces)

        for idx in range(cnt):
            surface = surfaces[idx] if type(surfaces) == type([]) else surfaces

            # Filter surface particles by object
            surface_obj1 = surface[object_labels_np[surface] == 1]
            surface_obj2 = surface[object_labels_np[surface] == 2]

            # Store counts
            self.sim_surface_cnt_obj1[idx] = len(surface_obj1)
            self.sim_surface_cnt_obj2[idx] = len(surface_obj2)

            # Load per-object surface indices
            if len(surface_obj1) > 0:
                self.load_sim_surface_obj(surface_obj1, len(surface_obj1), idx, obj_id=1)
            if len(surface_obj2) > 0:
                self.load_sim_surface_obj(surface_obj2, len(surface_obj2), idx, obj_id=2)

    @ti.kernel
    def load_sim_surface_obj(self, surface_index: ti.types.ndarray(), cnt: ti.int32, f: ti.int32, obj_id: ti.int32):
        for i in range(cnt):
            if obj_id == 1:
                self.sim_surface_index_obj1[f, i] = surface_index[i]
            else:  # obj_id == 2
                self.sim_surface_index_obj2[f, i] = surface_index[i]

    def compute_velocities(self):
        cnt = self.init_vol.shape[0]
        velocities = self.init_vel.repeat(cnt).reshape(cnt, -1)
        centroid = self.init_vol.sum(dim=0) / self.init_vol.shape[0]
        omega = self.init_omega.repeat(cnt).reshape(cnt, -1)
        velocities += torch.cross(omega, self.init_vol - centroid)
        return velocities

    def initialize(self):
        torch.cuda.synchronize()
        ti.sync()
        self.pos_grad_seq.clear()
        self.image_loss = 0.0
        self.per_object_sil_losses_accum = {1: 0.0, 2: 0.0}  # Reset accumulated per-object losses
        self.loss[None] = 0.0
        self.loss_obj1[None] = 0.0  # Reset per-object geometry loss for object 1
        self.loss_obj2[None] = 0.0  # Reset per-object geometry loss for object 2
        cnt = self.init_vol.shape[0]
        # Build per-particle velocities from per-object velocity parameters
        velocities = torch.zeros((cnt, 3), device=self.device)
        object_labels_np = self.object_labels.cpu().numpy() if torch.is_tensor(
            self.object_labels) else self.object_labels

        for obj_id_str, vel_param in self.object_velocities.items():
            obj_id = int(obj_id_str)
            obj_mask = object_labels_np == obj_id
            velocities[obj_mask] = vel_param.expand(obj_mask.sum(), -1)
        # velocities = self.compute_velocities()

        particles = self.init_vol
        self.dx[None], self.inv_dx[None] = self.voxel_size, 1.0 / self.voxel_size
        self.simulator.reset_dt()

        self.compute_particle_vol()
        # Build per-particle rho from object configs and make it trainable
        init_rhos_values = torch.zeros(cnt, device=particles.device)
        object_labels_np_for_rho = self.object_labels.cpu().numpy() if torch.is_tensor(
            self.object_labels) else self.object_labels
        for obj_config in self.args.sub_objects:
            obj_id = obj_config['object_id']
            obj_rho = obj_config.get('rho', 1000.0)
            obj_mask = object_labels_np_for_rho == obj_id
            init_rhos_values[obj_mask] = obj_rho

        # Make it trainable
        self.init_rhos = nn.Parameter(init_rhos_values)
        particle_rho = self.init_rhos

        self.clear_grads()

        # Build per-particle parameters from training_params dict
        particle_materials = self.particle_materials.cpu().numpy().astype(np.int32)
        object_labels_np = self.object_labels.cpu().numpy() if torch.is_tensor(
            self.object_labels) else self.object_labels

        self.init_mu = torch.zeros(cnt, device=particles.device)
        self.init_lam = torch.zeros(cnt, device=particles.device)
        self.init_yield_stress = torch.zeros(cnt, device=particles.device)
        self.init_plastic_viscosity = torch.zeros(cnt, device=particles.device)
        self.init_friction_alpha = torch.zeros(cnt, device=particles.device)
        cohesion = torch.zeros(cnt, device=particles.device)  # Not trainable

        # Fill parameters for each object
        for obj_id_str, obj_params in self.training_params.items():
            obj_id = int(obj_id_str)
            obj_mask = torch.from_numpy(object_labels_np == obj_id).to(particles.device)

            if not obj_mask.any():
                continue

            obj_material = obj_params['material']
            if obj_material == MPMSimulator.elasticity:
                E = (10 ** obj_params['E']).float()
                nu = constraint(obj_params['nu'], self.nu_bound).float()
                self.init_mu[obj_mask] = E / (2. * (1. + nu))
                self.init_lam[obj_mask] = E * nu / ((1. + nu) * (1. - 2. * nu))

            elif obj_material == MPMSimulator.von_mises:
                E = (10 ** obj_params['E']).float()
                nu = constraint(obj_params['nu'], self.nu_bound).float()
                self.init_mu[obj_mask] = E / (2. * (1. + nu))
                self.init_lam[obj_mask] = E * nu / ((1. + nu) * (1. - 2. * nu))
                self.init_yield_stress[obj_mask] = (10 ** obj_params['yield_stress']).float()

            elif obj_material == MPMSimulator.drucker_prager:
                E = (10 ** obj_params['E']).float()
                nu = constraint(obj_params['nu'], self.nu_bound).float()
                self.init_mu[obj_mask] = E / (2. * (1. + nu))
                self.init_lam[obj_mask] = E * nu / ((1. + nu) * (1. - 2. * nu))
                sin_phi = torch.sin(obj_params['friction_alpha'].float() / 180 * np.pi)
                self.init_friction_alpha[obj_mask] = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
                cohesion[obj_mask] = obj_params['cohesion'].float()

            elif obj_material == MPMSimulator.viscous_fluid:
                self.init_mu[obj_mask] = (10 ** obj_params['mu']).float()
                lam = (10 ** obj_params['kappa']).float() - 2. / 3. * (10 ** obj_params['mu']).float()
                self.init_lam[obj_mask] = lam

            elif obj_material == MPMSimulator.non_newtonian:
                self.init_mu[obj_mask] = (10 ** obj_params['mu']).float()
                lam = (10 ** obj_params['kappa']).float() - 2. / 3. * (10 ** obj_params['mu']).float()
                self.init_lam[obj_mask] = lam
                self.init_plastic_viscosity[obj_mask] = (10 ** obj_params['plastic_viscosity']).float()
                self.init_yield_stress[obj_mask] = (10 ** obj_params['yield_stress']).float()

        self.init_pos = self.init_vol.clone().requires_grad_(True)
        self.init_velocities = velocities
        self.simulator.cached_states.clear()
        self.from_torch(particles.data.cpu().numpy(), velocities.data.cpu().numpy(), particle_rho.data.cpu().numpy(),
                        self.init_mu.data.cpu().numpy(), self.init_lam.data.cpu().numpy(),
                        self.init_yield_stress.data.cpu().numpy(), self.init_plastic_viscosity.data.cpu().numpy(),
                        self.init_friction_alpha.data.cpu().numpy(), cohesion.data.cpu().numpy(),
                        particle_materials.astype(np.int32))
        self.compute_particle_mass()
        self.simulator.cfl_satisfy[None] = True
        torch.cuda.empty_cache()

    @ti.kernel
    def compute_particle_vol(self):
        grid_vol = (self.dx[None] * 0.5) ** 3
        for p in range(self.num_particles[None]):
            self.simulator.p_vol[p] = grid_vol

    def clear_grads(self):
        self.particle_rho.grad.fill(0)
        self.simulator.clear_grads()

    @ti.kernel
    def from_torch(self, particles: ti.types.ndarray(),
                   velocities: ti.types.ndarray(),
                   particle_rho: ti.types.ndarray(),
                   particle_mu: ti.types.ndarray(),
                   particle_lam: ti.types.ndarray(),
                   particle_yield_stress: ti.types.ndarray(),
                   particle_plastic_viscosity: ti.types.ndarray(),
                   particle_friction_alpha: ti.types.ndarray(),
                   particle_cohesion: ti.types.ndarray(),
                   particle_materials: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            self.particle_rho[p] = particle_rho[p]
            self.simulator.mu[p] = particle_mu[p]
            self.simulator.lam[p] = particle_lam[p]
            self.simulator.p_material[p] = particle_materials[p]
            self.simulator.yield_stress[p] = particle_yield_stress[p]
            self.simulator.plastic_viscosity[p] = particle_plastic_viscosity[p]
            self.simulator.friction_alpha[p] = particle_friction_alpha[p]
            self.simulator.cohesion[p] = particle_cohesion[p]
            self.simulator.p_mass[p] = 0.0
            self.simulator.F[p, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            for d in ti.static(range(3)):
                self.simulator.x[p, 0][d] = particles[p, d]
                self.simulator.v[p, 0][d] = velocities[p, d]

    def succeed(self):
        return self.simulator.cfl_satisfy[None]

    def get_per_object_losses(self):
        """Get per-object geometry losses for reporting"""
        return {
            1: self.loss_obj1[None],
            2: self.loss_obj2[None]
        }

    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            self.simulator.p_mass[p] = self.particle_rho[p] * self.simulator.p_vol[p]

    @ti.kernel
    def save_gt(self, f: ti.i32, gt: ti.types.ndarray(), count: ti.i32):
        for p in range(count):
            for d in ti.static(range(3)):
                self.gt[f, p][d] = gt[p, d]

    @ti.func
    def compute_distance(self, p1, p2):
        d = ti.math.distance(p1, p2)
        return d

    @ti.kernel
    def update_match_indices_gt2sim(self, f: ti.i32, local_index: ti.i32):
        err = 0.0
        self.gt2sim_err_cnt[f] = self.num_particles_surface[f]
        for i in range(self.num_particles_surface[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.sim_surface_cnt[f]):
                index_ = self.sim_surface_index[f, j]
                d = ti.math.distance(self.simulator.x[index_, local_index], self.gt[f, i])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = index_
            err += min_value
            self.match_indices_gt2sim[f, i] = min_index
            if self.gt[f, i].y <= self.voxel_size:
                self.gt2sim_err_cnt[f] -= 1

    @ti.kernel
    def update_match_indices_sim2gt(self, f: ti.i32, local_index: ti.i32):
        self.sim2gt_err_cnt[f] = self.sim_surface_cnt[f]
        for i in range(self.sim_surface_cnt[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.num_particles_surface[f]):
                d = ti.math.distance(self.simulator.x[self.sim_surface_index[f, i], local_index], self.gt[f, j])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = j
            self.match_indices_sim2gt[f, i] = min_index
            if not (self.gt[f, min_index].y > self.voxel_size and
                    self.simulator.x[self.sim_surface_index[f, i], local_index].y > self.voxel_size):
                self.sim2gt_err_cnt[f] -= 1

    # Per-object match indices update methods
    @ti.kernel
    def update_match_indices_gt2sim_obj1(self, f: ti.i32, local_index: ti.i32):
        self.gt2sim_err_cnt_obj1[f] = self.num_particles_surface_obj1[f]
        for i in range(self.num_particles_surface_obj1[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.sim_surface_cnt_obj1[f]):
                index_ = self.sim_surface_index_obj1[f, j]
                d = ti.math.distance(self.simulator.x[index_, local_index], self.gt_obj1[f, i])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = index_
            self.match_indices_gt2sim_obj1[f, i] = min_index
            if self.gt_obj1[f, i].y <= self.voxel_size:
                self.gt2sim_err_cnt_obj1[f] -= 1

    @ti.kernel
    def update_match_indices_sim2gt_obj1(self, f: ti.i32, local_index: ti.i32):
        self.sim2gt_err_cnt_obj1[f] = self.sim_surface_cnt_obj1[f]
        for i in range(self.sim_surface_cnt_obj1[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.num_particles_surface_obj1[f]):
                d = ti.math.distance(self.simulator.x[self.sim_surface_index_obj1[f, i], local_index], self.gt_obj1[f, j])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = j
            self.match_indices_sim2gt_obj1[f, i] = min_index
            if not (self.gt_obj1[f, min_index].y > self.voxel_size and
                    self.simulator.x[self.sim_surface_index_obj1[f, i], local_index].y > self.voxel_size):
                self.sim2gt_err_cnt_obj1[f] -= 1

    @ti.kernel
    def update_match_indices_gt2sim_obj2(self, f: ti.i32, local_index: ti.i32):
        self.gt2sim_err_cnt_obj2[f] = self.num_particles_surface_obj2[f]
        for i in range(self.num_particles_surface_obj2[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.sim_surface_cnt_obj2[f]):
                index_ = self.sim_surface_index_obj2[f, j]
                d = ti.math.distance(self.simulator.x[index_, local_index], self.gt_obj2[f, i])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = index_
            self.match_indices_gt2sim_obj2[f, i] = min_index
            if self.gt_obj2[f, i].y <= self.voxel_size:
                self.gt2sim_err_cnt_obj2[f] -= 1

    @ti.kernel
    def update_match_indices_sim2gt_obj2(self, f: ti.i32, local_index: ti.i32):
        self.sim2gt_err_cnt_obj2[f] = self.sim_surface_cnt_obj2[f]
        for i in range(self.sim_surface_cnt_obj2[f]):
            min_value, min_index = ti.math.inf, 0
            for j in range(self.num_particles_surface_obj2[f]):
                d = ti.math.distance(self.simulator.x[self.sim_surface_index_obj2[f, i], local_index], self.gt_obj2[f, j])
                old_min_value = ti.atomic_min(min_value, d)
                if min_value != old_min_value:
                    min_index = j
            self.match_indices_sim2gt_obj2[f, i] = min_index
            if not (self.gt_obj2[f, min_index].y > self.voxel_size and
                    self.simulator.x[self.sim_surface_index_obj2[f, i], local_index].y > self.voxel_size):
                self.sim2gt_err_cnt_obj2[f] -= 1

    @ti.kernel
    def compute_loss_gt2sim(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from gt to sim
        '''
        for i in range(self.num_particles_surface[f]):
            index_ = self.match_indices_gt2sim[f, i]
            d = self.compute_distance(self.simulator.x[index_, local_index], self.gt[f, i])

            if self.stage[None] == self.velocity_stage:
                self.loss[None] += d / self.num_particles_surface[f]
            else:
                # physical params stage
                self.loss[None] += d * self.w_geo[None] / self.gt2sim_err_cnt[f] if self.gt[
                                                                                        f, i].y > self.voxel_size else 0.0

    @ti.kernel
    def compute_loss_sim2gt(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from sim to gt
        '''
        # cnt = 0
        # loss = 0.0
        for i in range(self.sim_surface_cnt[f]):
            index_ = self.match_indices_sim2gt[f, i]
            x = self.simulator.x[self.sim_surface_index[f, i], local_index]
            d = self.compute_distance(x, self.gt[f, index_])
            if self.stage[None] == self.velocity_stage:
                self.loss[None] += d / self.sim_surface_cnt[f]
            else:
                # physical params stage
                self.loss[None] += d * self.w_geo[None] / self.sim2gt_err_cnt[f] if self.gt[
                                                                                        f, index_].y > self.voxel_size and x.y > self.voxel_size else 0.0

    # Per-object loss computation methods
    @ti.kernel
    def compute_loss_gt2sim_obj1(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from gt to sim for object 1
        '''
        for i in range(self.num_particles_surface_obj1[f]):
            index_ = self.match_indices_gt2sim_obj1[f, i]
            d = self.compute_distance(self.simulator.x[index_, local_index], self.gt_obj1[f, i])

            if self.stage[None] == self.velocity_stage:
                self.loss_obj1[None] += d / self.num_particles_surface_obj1[f]
            else:
                # physical params stage
                self.loss_obj1[None] += d * self.w_geo[None] / self.gt2sim_err_cnt_obj1[f] if self.gt_obj1[
                                                                                        f, i].y > self.voxel_size else 0.0

    @ti.kernel
    def compute_loss_sim2gt_obj1(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from sim to gt for object 1
        '''
        for i in range(self.sim_surface_cnt_obj1[f]):
            index_ = self.match_indices_sim2gt_obj1[f, i]
            x = self.simulator.x[self.sim_surface_index_obj1[f, i], local_index]
            d = self.compute_distance(x, self.gt_obj1[f, index_])
            if self.stage[None] == self.velocity_stage:
                self.loss_obj1[None] += d / self.sim_surface_cnt_obj1[f]
            else:
                # physical params stage
                self.loss_obj1[None] += d * self.w_geo[None] / self.sim2gt_err_cnt_obj1[f] if self.gt_obj1[
                                                                                        f, index_].y > self.voxel_size and x.y > self.voxel_size else 0.0

    @ti.kernel
    def compute_loss_gt2sim_obj2(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from gt to sim for object 2
        '''
        for i in range(self.num_particles_surface_obj2[f]):
            index_ = self.match_indices_gt2sim_obj2[f, i]
            d = self.compute_distance(self.simulator.x[index_, local_index], self.gt_obj2[f, i])

            if self.stage[None] == self.velocity_stage:
                self.loss_obj2[None] += d / self.num_particles_surface_obj2[f]
            else:
                # physical params stage
                self.loss_obj2[None] += d * self.w_geo[None] / self.gt2sim_err_cnt_obj2[f] if self.gt_obj2[
                                                                                        f, i].y > self.voxel_size else 0.0

    @ti.kernel
    def compute_loss_sim2gt_obj2(self, f: ti.i32, local_index: ti.i32):
        '''
        compute loss: from sim to gt for object 2
        '''
        for i in range(self.sim_surface_cnt_obj2[f]):
            index_ = self.match_indices_sim2gt_obj2[f, i]
            x = self.simulator.x[self.sim_surface_index_obj2[f, i], local_index]
            d = self.compute_distance(x, self.gt_obj2[f, index_])
            if self.stage[None] == self.velocity_stage:
                self.loss_obj2[None] += d / self.sim_surface_cnt_obj2[f]
            else:
                # physical params stage
                self.loss_obj2[None] += d * self.w_geo[None] / self.sim2gt_err_cnt_obj2[f] if self.gt_obj2[
                                                                                        f, index_].y > self.voxel_size and x.y > self.voxel_size else 0.0

    def get_surface(self, f):
        surface = np.full((self.num_particles[None], 3), [0, 255, 0], dtype=np.uint8)

        @ti.kernel
        def extract(f: ti.i32, surface: ti.types.ndarray()):
            local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
            for i in range(self.sim_surface_cnt[f]):
                index = self.sim_surface_index[f, i]
                surface[index, 0] = ti.cast(255, ti.u8)
                surface[index, 1] = ti.cast(0, ti.u8)
                surface[index, 2] = ti.cast(0, ti.u8)
                # for d in ti.static(range(3)):
                # surface[i, d] = ti.cast(self.simulator.x[index, local_index][d], ti.f32)

        extract(f, surface)
        return surface

    def get_surface_vertics(self, f):
        surface = np.zeros([self.sim_surface_cnt[f], 3], dtype=np.float32)
        color = np.full((self.sim_surface_cnt[f], 3), [0, 255, 0], dtype=np.uint8)

        @ti.kernel
        def extract(f: ti.i32, surface: ti.types.ndarray(), color: ti.types.ndarray()):
            local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
            for i in range(self.sim_surface_cnt[f]):
                index = self.sim_surface_index[f, i]
                for d in ti.static(range(3)):
                    surface[i, d] = ti.cast(self.simulator.x[index, local_index][d], ti.f32)
            for i in range(self.sim_surface_cnt[f]):
                if surface[i, 1] <= self.voxel_size:
                    color[i, 0] = ti.cast(0, ti.u8)
                    color[i, 1] = ti.cast(0, ti.u8)
                    color[i, 2] = ti.cast(255, ti.u8)

        extract(f, surface, color)
        return surface, color

    def render_forward(self, f, xyz, backward=True):
        gaussians = self.scene.gaussians
        views = self.views[f]
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        d_xyz = xyz - gaussians.get_xyz
        loss_img = torch.tensor(0.0, device=self.device)
        loss_alp = torch.tensor(0.0, device=self.device)

        # Check if we have per-object GT masks and object probabilities
        has_per_object = (hasattr(gaussians, 'get_object_probs') and
                          len(views) > 0 and
                          hasattr(views[0], 'gt_alpha_mask_obj1'))

        if has_per_object:
            # Per-object silhouette loss
            loss_alp_obj1 = torch.tensor(0.0, device=self.device)
            loss_alp_obj2 = torch.tensor(0.0, device=self.device)

            # Get object assignments using argmax (same as filter_gaussians_by_object)
            object_probs = gaussians.get_object_probs  # [N, 3]
            object_assignments = torch.argmax(object_probs, dim=1)  # [N] with values 0,1,2
            obj1_mask = (object_assignments == 1).float()  # Mask for object 1
            obj2_mask = (object_assignments == 2).float()  # Mask for object 2

            # Store original opacities (raw _opacity values)
            original_opacity = gaussians._opacity.detach().clone()

            # Create zero deformation tensors with proper shape
            # Isotropic Gaussians (num_attribute=4): use scalar 0.0 for both
            d_scaling = 0.0
            d_rotation = 0.0
            # Anisotropic Gaussians (num_attribute=10): d_scaling shape [N, 3], d_rotation shape [N, 4]
            # num_gaussians = gaussians._xyz.shape[0]
            # d_scaling = torch.zeros((num_gaussians, 3), device=gaussians._xyz.device)
            # d_rotation = torch.zeros((num_gaussians, 4), device=gaussians._xyz.device)

            for idx, view in enumerate(views):
                # Render object 1 only
                gaussians._opacity.data = original_opacity * obj1_mask.unsqueeze(1) - 1e2 * (1 - obj1_mask.unsqueeze(1))
                # Anisotropic version (current)
                results_obj1 = render(view, gaussians, self.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                # Isotropic version (fallback)
                # results_obj1 = render(view, gaussians, self.pipeline, background, d_xyz, 0.0, 0.0, False)
                alpha_obj1 = results_obj1["alpha"]
                gt_alpha_mask_obj1 = view.gt_alpha_mask_obj1

                # Render object 2 only
                gaussians._opacity.data = original_opacity * obj2_mask.unsqueeze(1) - 1e2 * (1 - obj2_mask.unsqueeze(1))
                # Anisotropic version (current)
                results_obj2 = render(view, gaussians, self.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                # Isotropic version (fallback)
                # results_obj2 = render(view, gaussians, self.pipeline, background, d_xyz, 0.0, 0.0, False)
                alpha_obj2 = results_obj2["alpha"]
                gt_alpha_mask_obj2 = view.gt_alpha_mask_obj2

                # Compute per-object silhouette losses
                if self.w_alp > 0.0:
                    # Object 1 loss
                    loss_alp_obj1 += l1_loss(alpha_obj1, gt_alpha_mask_obj1)
                    # Object 2 loss
                    loss_alp_obj2 += l1_loss(alpha_obj2, gt_alpha_mask_obj2)

                # Also compute image loss with full opacity (original behavior)
                if self.w_img > 0.0:
                    gaussians._opacity.data = original_opacity
                    # Anisotropic version (current)
                    results = render(view, gaussians, self.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                    # Isotropic version (fallback)
                    # results = render(view, gaussians, self.pipeline, background, d_xyz, 0.0, 0.0, False)
                    image = results["render"]
                    gt_image = view.original_image.cuda()
                    # Crop image
                    mask = torch.logical_or(gt_image.sum(0) != 0, image.sum(0) != 0)
                    ids = torch.where(mask)
                    h_min, h_max = max(ids[0].min() - 50, 0), min(ids[0].max() + 50, image.shape[1])
                    w_min, w_max = max(ids[1].min() - 50, 0), min(ids[1].max() + 50, image.shape[2])
                    image = image[:, h_min:h_max, w_min:w_max]
                    gt_image = gt_image[:, h_min:h_max, w_min:w_max]
                    Ll1 = l1_loss(image, gt_image)
                    loss_img += (1.0 - self.image_op.lambda_dssim) * Ll1 + self.image_op.lambda_dssim * (
                                1.0 - ssim(image, gt_image))

            # Restore original opacities
            gaussians._opacity.data = original_opacity

            # Combine losses: sum of per-object silhouette losses
            loss_alp = loss_alp_obj1 + loss_alp_obj2
            loss = (self.w_img * loss_img + self.w_alp * loss_alp) / len(views)

            # Accumulate per-object losses for logging
            frame_losses = {
                1: (loss_alp_obj1 / len(views)).item() if len(views) > 0 else 0.0,
                2: (loss_alp_obj2 / len(views)).item() if len(views) > 0 else 0.0
            }
            self.per_object_sil_losses_accum[1] += frame_losses[1]
            self.per_object_sil_losses_accum[2] += frame_losses[2]
            # Store current frame losses for immediate access if needed
            self.per_object_sil_losses = frame_losses
        else:
            # Original combined silhouette loss (fallback)
            # Create zero deformation tensors with proper shape
            # Isotropic Gaussians (num_attribute=4): use scalar 0.0 for both
            d_scaling = 0.0
            d_rotation = 0.0
            # Anisotropic Gaussians (num_attribute=10): d_scaling shape [N, 3], d_rotation shape [N, 4]
            # num_gaussians = gaussians._xyz.shape[0]
            # d_scaling = torch.zeros((num_gaussians, 3), device=gaussians._xyz.device)
            # d_rotation = torch.zeros((num_gaussians, 4), device=gaussians._xyz.device)

            for view in views:
                # Anisotropic version (current)
                results = render(view, gaussians, self.pipeline, background, d_xyz, d_rotation, d_scaling, False)
                # Isotropic version (fallback)
                # results = render(view, gaussians, self.pipeline, background, d_xyz, 0.0, 0.0, False)
                image, alpha = results["render"], results["alpha"]
                gt_image = view.original_image.cuda()
                gt_alpha_mask = view.gt_alpha_mask
                # crop image loss
                mask = torch.logical_or(gt_image.sum(0) != 0,
                                        image.sum(0) != 0)
                ids = torch.where(mask)
                h_min, h_max, w_min, w_max = ids[0].min(), ids[0].max(), ids[1].min(), ids[1].max()
                h_min, h_max = max(h_min - 50, 0), min(h_max + 50, image.shape[1])
                w_min, w_max = max(w_min - 50, 0), min(w_max + 50, image.shape[2])
                image = image[:, h_min:h_max, w_min:w_max]
                gt_image = gt_image[:, h_min:h_max, w_min:w_max]
                alpha = alpha[:, h_min:h_max, w_min:w_max]
                gt_alpha_mask = gt_alpha_mask[:, h_min:h_max, w_min:w_max]

                if self.w_img > 0.0:
                    Ll1 = l1_loss(image, gt_image)
                    loss_img += (1.0 - self.image_op.lambda_dssim) * Ll1 + self.image_op.lambda_dssim * (
                            1.0 - ssim(image, gt_image))
                if self.w_alp > 0.0:
                    L_alpha = l1_loss(alpha, gt_alpha_mask)
                    loss_alp += L_alpha
            loss = (self.w_img * loss_img + self.w_alp * loss_alp) / len(views)

        # Ensure loss has gradient even when it's zero
        if not loss.requires_grad and backward:
            # Create a dummy loss with gradient to avoid backward error
            loss = loss + 0.0 * xyz.sum() * 0.0

        if backward:
            if loss.requires_grad:
                loss.backward()
        with torch.no_grad():
            self.image_loss += loss.detach().cpu()
        self.pos_grad_seq.append(xyz.grad)

    def forward(self, f, img_backward=True):
        particle_pos = np.zeros([self.num_particles[None], 3], dtype=np.float32)

        if f > 0:
            self.simulator.advance(f - 1)

        if not self.succeed():
            return particle_pos
        local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size

        if len(self.gts) > 0 and (self.geo_loss or self.stage[None] == self.velocity_stage):
            # Check if we should use per-object geometry loss
            use_per_object_geo_loss = (self.gts_per_object is not None and
                                        self.object_labels is not None and
                                        self.sim_surface_cnt_obj1[f] > 0 and
                                        self.sim_surface_cnt_obj2[f] > 0)

            if use_per_object_geo_loss:
                # Per-object loss computation
                # Object 1
                self.update_match_indices_gt2sim_obj1(f, local_index)
                self.update_match_indices_sim2gt_obj1(f, local_index)
                self.compute_loss_gt2sim_obj1(f, local_index)
                self.compute_loss_sim2gt_obj1(f, local_index)

                # Object 2
                self.update_match_indices_gt2sim_obj2(f, local_index)
                self.update_match_indices_sim2gt_obj2(f, local_index)
                self.compute_loss_gt2sim_obj2(f, local_index)
                self.compute_loss_sim2gt_obj2(f, local_index)

                # Combine per-object losses (equal weighting for now)
                self.loss[None] = self.loss_obj1[None] + self.loss_obj2[None]
            else:
                # Original combined loss computation
                self.update_match_indices_gt2sim(f, local_index)
                self.update_match_indices_sim2gt(f, local_index)
                self.compute_loss_gt2sim(f, local_index)
                self.compute_loss_sim2gt(f, local_index)

        self.simulator.get_x(f, particle_pos)
        particle_pos = torch.from_numpy(particle_pos).to(self.device).requires_grad_()
        if (f > 0 and self.stage[None] == self.physical_params_stage and self.img_loss) or \
                (f > 0 and self.stage[None] == self.velocity_stage and len(self.gts) == 0):
            self.render_forward(f, particle_pos, img_backward)
        return particle_pos

    @ti.kernel
    def set_pos_grad(self, f: ti.i32, dLdpo: ti.types.ndarray()):
        s = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
        for p in range(self.num_particles[None]):
            for d in ti.static(range(3)):
                self.simulator.x.grad[p, s][d] += dLdpo[p, d]

    def backward(self, f):
        local_index = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size

        if self.stage[None] == self.physical_params_stage and f > 0 and self.img_loss or (len(self.gts) == 0):
            self.set_pos_grad(f, self.pos_grad_seq[f - 1].data.cpu().numpy())
        if self.geo_loss or self.stage[None] == self.velocity_stage:
            # Check if we're using per-object geometry loss
            use_per_object_geo_loss = (self.gts_per_object is not None and
                                        self.object_labels is not None and
                                        self.sim_surface_cnt_obj1[f] > 0 and
                                        self.sim_surface_cnt_obj2[f] > 0)

            if use_per_object_geo_loss:
                # Per-object backward pass
                # Since loss[None] = loss_obj1[None] + loss_obj2[None],
                # the gradient of each component is 1.0
                self.loss_obj1.grad[None] = 1.0
                self.loss_obj2.grad[None] = 1.0

                self.compute_loss_sim2gt_obj1.grad(f, local_index)
                self.compute_loss_gt2sim_obj1.grad(f, local_index)
                self.compute_loss_sim2gt_obj2.grad(f, local_index)
                self.compute_loss_gt2sim_obj2.grad(f, local_index)
            else:
                # Original combined backward pass
                self.compute_loss_sim2gt.grad(f, local_index)
                self.compute_loss_gt2sim.grad(f, local_index)
        if f > 0:
            self.simulator.advance_grad(f - 1)
        else:
            self.compute_particle_mass.grad()
            dtype = np.float32
            velocity_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            position_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            rho_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            mu_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            lam_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            # Now per-particle gradients
            yield_stress_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            viscosity_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            friction_alpha_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            cohesion_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            self.get_input_grad(position_grad, velocity_grad, rho_grad, mu_grad, lam_grad,
                                yield_stress_grad, viscosity_grad, friction_alpha_grad, cohesion_grad)
            return torch.from_numpy(position_grad).to(self.device), \
                torch.from_numpy(velocity_grad).to(self.device), \
                torch.from_numpy(mu_grad).to(self.device), torch.from_numpy(lam_grad).to(self.device), \
                torch.from_numpy(yield_stress_grad).to(self.device), torch.from_numpy(viscosity_grad).to(self.device), \
                torch.from_numpy(friction_alpha_grad).to(self.device), torch.from_numpy(cohesion_grad).to(self.device), \
                torch.from_numpy(rho_grad).to(self.device)
