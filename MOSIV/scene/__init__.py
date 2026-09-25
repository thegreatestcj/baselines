#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import copy
import json
import torch
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], pcd=None, load_fix_pcd=False, cam_info=None, object_id=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {} if cam_info is None else cam_info.get("train_cams")
        self.test_cameras = {} if cam_info is None else cam_info.get("test_cams")
        read_cam = True if cam_info is None else False
        if not read_cam:
            print('use given cams')
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "all_data.json")):
            # Check if this is GenesisMO format (has metadata.json) or PAC-NeRF format
            if os.path.exists(os.path.join(args.source_path, "metadata.json")) or "GenesisMO" in args.source_path:
                print("Found metadata.json or GenesisMO in path, assuming GenesisMO multi-object data set!")
                scene_info = sceneLoadTypeCallbacks["GenesisMO"](args.source_path, args.config_path, args.white_background, load_fix_pcd=load_fix_pcd, read_cam=read_cam)
            else:
                print("Found all_data.json file, assuming PacNeRF data set!")
                scene_info = sceneLoadTypeCallbacks["PacNeRF"](args.source_path, args.config_path, args.white_background, load_fix_pcd=load_fix_pcd, read_cam=read_cam)
        elif os.path.exists(os.path.join(args.source_path, "camera.json")) and os.path.exists(os.path.join(args.source_path, "frame.json")):
            print("Found all_data.json file, assuming SpringGaus MPM Synthetic data set!")
            scene_info = sceneLoadTypeCallbacks["SpringGausMPMSynthetic"](args.source_path, args.config_path, args.white_background, args.num_frame)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif 'real_capture' in args.source_path:
            print("Found real_capture, assuming Spring-Gaus Real Capture data set!")
            scene_info = sceneLoadTypeCallbacks["SpringGausRealCapture"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and read_cam:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle and read_cam:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"] if read_cam else cam_info.get("cameras_extent")

        for resolution_scale in resolution_scales:
            if not read_cam:
                continue
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        if pcd is not None:
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)
        else:
            if self.loaded_iter:
                if not load_fix_pcd:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"),
                                            og_number_points=len(scene_info.point_cloud.points))
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud_fix_pcd",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, fix_pcd=False):
        name = "point_cloud/iteration_{}".format(iteration) if not fix_pcd else "point_cloud_fix_pcd/iteration_{}".format(iteration)
        point_cloud_path = os.path.join(self.model_path, name)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def clipTrainCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.train_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.train_cameras = new_cameras

    def clipTestCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.test_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.test_cameras = new_cameras

    
    def overwrite_alphas(self, pipeline, dataset: ModelParams, deform: DeformModel, object_id=None):
        from gaussian_renderer import render
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        xyz_canonical = self.gaussians.get_xyz.detach()
        
        # Check if we have object probabilities for per-object rendering
        has_object_probs = hasattr(self.gaussians, 'get_object_probs')
        if has_object_probs:
            print("Rendering per-object alpha masks using object probabilities")
            object_probs = self.gaussians.get_object_probs  # [N, 3] with softmax probabilities
            # Use argmax to assign each Gaussian to exactly one object (same logic as filter_gaussians_by_object)
            object_assignments = torch.argmax(object_probs, dim=1)  # [N] with values 0,1,2
            obj_masks = {k: (object_assignments == k).float() for k in range(1, object_probs.shape[1])}  # per object id
            
            # Store original opacities (raw _opacity values)
            original_opacity = self.gaussians._opacity.clone().detach()
        
        def overwrite(cam_dict):
            for scale, cam_list in cam_dict.items():
                for view in cam_list:
                    fid = view.fid
                    time_input = fid.unsqueeze(0).expand(1, -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
                    
                    if has_object_probs:
                        # Render combined alpha mask (original behavior)
                        results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                        alpha = results["alpha"]
                        view.gt_alpha_mask = alpha.to(view.data_device)
                        # Generate the per-object mask of object_id (baselines: any number of objects)
                        if object_id is not None and int(object_id) in obj_masks:
                            m = obj_masks[int(object_id)]
                            self.gaussians._opacity.data = original_opacity * m.unsqueeze(1)
                            results_obj = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                            setattr(view, f'gt_alpha_mask_obj{int(object_id)}', results_obj["alpha"].to(view.data_device))
                            self.gaussians._opacity.data = original_opacity
                    else:
                        # Fallback to original behavior if no object probabilities
                        results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                        alpha = results["alpha"]
                        view.gt_alpha_mask = alpha.to(view.data_device)

        overwrite(self.train_cameras)
        overwrite(self.test_cameras)
        copy.deepcopy(self.cameras_extent)
        return self.train_cameras, self.test_cameras, self.cameras_extent
    
    def overwrite_alphas_with_pcds(
        self,
        pipeline,
        dataset: ModelParams,
        gt_pcs_per_object: dict,
        object_id: int,
    ):
        """
        假设：
        - 外部已经根据 object_id 把 self.gaussians 过滤好了
            比如：
                gaussians_obj1 = filter_gaussians_by_object(original_gaussians, object_id=1, ...)
                scene_full.gaussians = gaussians_obj1
                scene_full.overwrite_alphas_with_pcds(pipeline, dataset, gt_pcs_per_object, object_id=1)

        - gt_pcs_per_object[object_id] 是一个长度为 T 的列表，每个元素是 [N_obj, 3] 的 GT 点云
        - self.gaussians.get_xyz 对应的 canonical xyz 也是 [N_obj, 3]
        作用：
        - 对 train/test cameras，根据 view.fid 选帧，算出 d_xyz，用当前 object 的 Gaussians 渲染 alpha
        - 把 alpha 写进：
                object_id == 1 -> view.gt_alpha_mask_obj1
                object_id == 2 -> view.gt_alpha_mask_obj2
                其他 -> view.gt_alpha_mask
        """
        from gaussian_renderer import render
        import torch

        assert object_id is not None, "overwrite_alphas_with_pcds 需要指定 object_id"

        device = self.gaussians.get_xyz.device
        xyz_canonical = self.gaussians.get_xyz.detach()     # [N_obj, 3]
        N_obj = xyz_canonical.shape[0]

        # ---- 1. 取出当前 object 的 GT 点云序列 ----
        assert object_id in gt_pcs_per_object, \
            f"gt_pcs_per_object 中不包含 object_id={object_id}，keys={list(gt_pcs_per_object.keys())}"

        gt_frames = gt_pcs_per_object[object_id]   # list of [N_obj, 3]
        T = len(gt_frames)
        assert T > 0, "当前 object 的 GT 点云帧数为 0"

        # 检查帧 0 的点数是否和 Gaussians 数量对齐
        assert gt_frames[0].shape[0] == N_obj, \
            f"Gaussians 数量 {N_obj} 和 GT 点云数量 {gt_frames[0].shape[0]} 对不上"

        # ---- 2. 背景色 & fid -> 帧号 映射 ----
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)


        fps = 24

        def frame_idx_from_fid(fid):
            if torch.is_tensor(fid):
                t = float(fid.item())
            else:
                t = float(fid)
            idx = int(round(t * fps))        # 关键是 round，而不是直接 int(t * fps)
            idx = max(0, min(idx, T - 1))    # 防止越界
            return idx
        
        
        
        # ---- 3. 对一组 camera 覆盖 alpha（只用当前 object 的 Gaussians，不做 opacity mask）----
        def overwrite(cam_dict):
            for scale, cam_list in cam_dict.items():
                for idx, view in enumerate(cam_list):
                    
                    frame_idx = frame_idx_from_fid(view.fid)
                    print(f"[object {object_id}] Rendering alpha for view fid={view.fid} -> frame_idx={frame_idx}")

                    gt_t = gt_frames[frame_idx].to(device)  # [N_obj, 3]
                    assert gt_t.shape[0] == N_obj, \
                        f"某帧 GT 点云数量变了: object {object_id} = {gt_t.shape[0]}, 期望 {N_obj}"

                    # 用 GT 位置减去 canonical 位置得到 d_xyz
                    d_xyz = gt_t - xyz_canonical              # [N_obj,3]
                    d_rotation = 0.0
                    d_scaling = 0.0

                    with torch.no_grad():
                        results = render(
                            view,
                            self.gaussians,
                            pipeline,
                            background,
                            d_xyz,
                            d_rotation,
                            d_scaling,
                            False
                        )
                    alpha = results["alpha"]

                    # 根据 object_id 写到对应字段
                    if object_id == 1:
                        view.gt_alpha_mask_obj1 = alpha.to(view.data_device)
                    elif object_id == 2:
                        view.gt_alpha_mask_obj2 = alpha.to(view.data_device)
                    else:
                        # 如果以后有更多 object，可以按需扩展
                        view.gt_alpha_mask = alpha.to(view.data_device)

        # 对 train / test cameras 都覆盖
        overwrite(self.train_cameras)
        overwrite(self.test_cameras)
        copy.deepcopy(self.cameras_extent)
        return self.train_cameras, self.test_cameras, self.cameras_extent