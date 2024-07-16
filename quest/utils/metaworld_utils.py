import copy
from collections import OrderedDict

import numpy as np
import quest.utils.file_utils as FileUtils
import quest.utils.obs_utils as ObsUtils
from PIL import Image
from quest.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
from quest.utils.frame_stack import FrameStackObservationFixed
import torch
import torch.nn as nn
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from quest.utils.mujoco_point_cloud import posRotMat2Mat, quat2Mat
import math
import mujoco
import os
from torch.utils.data import ConcatDataset
import pytorch3d.ops as torch3d_ops
import metaworld

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from quest.utils.draw_utils import aggr_point_cloud_from_data, show_point_cloud


from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)


_policies = OrderedDict(
    [
        ("assembly-v2", SawyerAssemblyV2Policy),
        ("basketball-v2", SawyerBasketballV2Policy),
        ("bin-picking-v2", SawyerBinPickingV2Policy),
        ("box-close-v2", SawyerBoxCloseV2Policy),
        ("button-press-topdown-v2", SawyerButtonPressTopdownV2Policy),
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallV2Policy),
        ("button-press-v2", SawyerButtonPressV2Policy),
        ("button-press-wall-v2", SawyerButtonPressWallV2Policy),
        ("coffee-button-v2", SawyerCoffeeButtonV2Policy),
        ("coffee-pull-v2", SawyerCoffeePullV2Policy),
        ("coffee-push-v2", SawyerCoffeePushV2Policy),
        ("dial-turn-v2", SawyerDialTurnV2Policy),
        ("disassemble-v2", SawyerDisassembleV2Policy),
        ("door-close-v2", SawyerDoorCloseV2Policy),
        ("door-lock-v2", SawyerDoorLockV2Policy),
        ("door-open-v2", SawyerDoorOpenV2Policy),
        ("door-unlock-v2", SawyerDoorUnlockV2Policy),
        ("drawer-close-v2", SawyerDrawerCloseV2Policy),
        ("drawer-open-v2", SawyerDrawerOpenV2Policy),
        ("faucet-close-v2", SawyerFaucetCloseV2Policy),
        ("faucet-open-v2", SawyerFaucetOpenV2Policy),
        ("hammer-v2", SawyerHammerV2Policy),
        ("hand-insert-v2", SawyerHandInsertV2Policy),
        ("handle-press-side-v2", SawyerHandlePressSideV2Policy),
        ("handle-press-v2", SawyerHandlePressV2Policy),
        ("handle-pull-v2", SawyerHandlePullV2Policy),
        ("handle-pull-side-v2", SawyerHandlePullSideV2Policy),
        ("peg-insert-side-v2", SawyerPegInsertionSideV2Policy),
        ("lever-pull-v2", SawyerLeverPullV2Policy),
        ("peg-unplug-side-v2", SawyerPegUnplugSideV2Policy),
        ("pick-out-of-hole-v2", SawyerPickOutOfHoleV2Policy),
        ("pick-place-v2", SawyerPickPlaceV2Policy),
        ("pick-place-wall-v2", SawyerPickPlaceWallV2Policy),
        ("plate-slide-back-side-v2", SawyerPlateSlideBackSideV2Policy),
        ("plate-slide-back-v2", SawyerPlateSlideBackV2Policy),
        ("plate-slide-side-v2", SawyerPlateSlideSideV2Policy),
        ("plate-slide-v2", SawyerPlateSlideV2Policy),
        ("reach-v2", SawyerReachV2Policy),
        ("reach-wall-v2", SawyerReachWallV2Policy),
        ("push-back-v2", SawyerPushBackV2Policy),
        ("push-v2", SawyerPushV2Policy),
        ("push-wall-v2", SawyerPushWallV2Policy),
        ("shelf-place-v2", SawyerShelfPlaceV2Policy),
        ("soccer-v2", SawyerSoccerV2Policy),
        ("stick-pull-v2", SawyerStickPullV2Policy),
        ("stick-push-v2", SawyerStickPushV2Policy),
        ("sweep-into-v2", SawyerSweepIntoV2Policy),
        ("sweep-v2", SawyerSweepV2Policy),
        ("window-close-v2", SawyerWindowCloseV2Policy),
        ("window-open-v2", SawyerWindowOpenV2Policy),
    ]
)
_env_names = list(_policies)

classes = {
    'ML45': {
        'train': ['assembly-v2', 
                  'basketball-v2', 
                  'button-press-topdown-v2', 
                  'button-press-topdown-wall-v2', 
                  'button-press-v2', 
                  'button-press-wall-v2', 
                  'coffee-button-v2', 
                  'coffee-pull-v2', 
                  'coffee-push-v2', 
                  'dial-turn-v2', 
                  'disassemble-v2', 
                  'door-close-v2', 
                  'door-open-v2', 
                  'drawer-close-v2', 
                  'drawer-open-v2', 
                  'faucet-close-v2', 
                  'faucet-open-v2', 
                  'hammer-v2', 
                  'handle-press-side-v2', 
                  'handle-press-v2', 
                  'handle-pull-side-v2', 
                  'handle-pull-v2', 
                  'lever-pull-v2', 
                  'peg-insert-side-v2', 
                  'peg-unplug-side-v2', 
                  'pick-out-of-hole-v2', 
                  'pick-place-v2', 
                  'pick-place-wall-v2', 
                  'plate-slide-back-side-v2', 
                  'plate-slide-back-v2', 
                  'plate-slide-side-v2', 
                  'plate-slide-v2', 
                  'push-back-v2', 
                  'push-v2', 
                  'push-wall-v2', 
                  'reach-v2', 
                  'reach-wall-v2', 
                  'shelf-place-v2', 
                  'soccer-v2', 
                  'stick-pull-v2', 
                  'stick-push-v2', 
                  'sweep-into-v2', 
                  'sweep-v2', 
                  'window-close-v2', 
                  'window-open-v2'],
        'test': ['bin-picking-v2', 
                 'box-close-v2', 
                 'door-lock-v2', 
                 'door-unlock-v2', 
                 'hand-insert-v2']
    },
    'ML45_PRISE': {
        'train': [
            'assembly-v2',
            'basketball-v2',
            'bin-picking-v2',
            'button-press-topdown-v2',
            'button-press-topdown-wall-v2',
            'button-press-v2',
            'button-press-wall-v2',
            'coffee-button-v2',
            'coffee-pull-v2',
            'coffee-push-v2',
            'dial-turn-v2',
            'door-close-v2',
            'door-lock-v2',
            'door-open-v2',
            'door-unlock-v2',
            'drawer-close-v2',
            'drawer-open-v2',
            'faucet-close-v2',
            'faucet-open-v2',
            'hammer-v2',
            'handle-press-side-v2',
            'handle-press-v2',
            'handle-pull-side-v2',
            'handle-pull-v2',
            'lever-pull-v2',
            'peg-insert-side-v2',
            'peg-unplug-side-v2',
            'pick-out-of-hole-v2',
            'pick-place-v2',
            'plate-slide-back-side-v2',
            'plate-slide-back-v2',
            'plate-slide-side-v2',
            'plate-slide-v2',
            'push-back-v2',
            'push-v2',
            'push-wall-v2',
            'reach-v2',
            'reach-wall-v2',
            'shelf-place-v2',
            'soccer-v2',
            'stick-push-v2',
            'sweep-into-v2',
            'sweep-v2',
            'window-close-v2',
            'window-open-v2'],
        'test': [
            'box-close-v2',
            'disassemble-v2',
            'hand-insert-v2',
            'pick-place-wall-v2',
            'stick-pull-v2',
        ]

    }
}

def get_index(env_name):
    return _env_names.index(env_name)

def get_expert():
    env_experts = {
        env_name: _policies[env_name]() for env_name in _policies
    }

    def expert(obs, task_id):
        obs_gt = obs['obs_gt'].squeeze()
        return env_experts[_env_names[task_id]].get_action(obs_gt)
    
    return expert

def get_hacked_task_id(task_id):
    # correspondences = [0,27,41,28,40]
    correspondences = {
        3: 0, # box is similar to assembly
        12: 30, # disassemble is similar to pick out of hole
        22: 46, # hand insert is similar to sweep into
        32: 31, # pick-place-wall is similar to pick-place
        44: 45, # stick-pull is similar to stick-push
    }
    if task_id in correspondences:
        return correspondences[task_id]
    else:
        return task_id

def get_env_expert(env_name):
    return _policies[env_name]()

def get_benchmark(benchmark_name):
    benchmarks = {
        'ML1': metaworld.ML1,
        'ML10': metaworld.ML10,
        'ML45': metaworld.ML45,
        'ML45_PRISE': ML45PRISEBenchmark,
    }
    return benchmarks[benchmark_name]()



class MetaWorldFrameStack(FrameStackObservationFixed):
    def __init__(self, 
                 env_name,
                 env_factory,
                 num_stack,
                 ):
        self.num_stack = num_stack
        
        env = env_factory(env_name)
        super().__init__(env, num_stack)

    def set_task(self, task):
        self.env.set_task(task)


def get_env_names(benchmark=None, mode=None):
    if benchmark is None:
        return list(_env_names)
    
    if type(benchmark) is str:
        return classes[benchmark][mode]
    else:
        env_names = list(benchmark.train_classes \
            if mode == 'train' else benchmark.test_classes)
        env_names.sort()
        return env_names
    
def get_tasks(benchmark, mode):
    if benchmark is None:
        return []
    return benchmark.train_tasks if mode == 'train' else benchmark.test_tasks


class MetaWorldWrapper(gymnasium.Wrapper):
    def __init__(self, 
                 env_name: str,
                 img_height: int = 128,
                 img_width: int = 128,
                 max_episode_length=500,
                 camera_name='corner2',
                 env_kwargs=None,):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](**env_kwargs)
        env._freeze_rand_vec = False
        env.max_path_length = max_episode_length
        super().__init__(env)
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self.camera_name = camera_name

        self.viewer = OffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        self.observation_space = gymnasium.spaces.Dict({
            'robot_states': gymnasium.spaces.Box(
                low=np.concatenate((self.env.observation_space.low[:4], self.env.observation_space.low[18:22])).astype(np.float32),
                high=np.concatenate((self.env.observation_space.high[:4], self.env.observation_space.high[18:22])).astype(np.float32),
                # dtype=np.float32
            ),
            'corner_rgb': gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            ),
            'obs_gt': self.env.observation_space
        })

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = super().step(action)
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt


        image_obs = self.render(mode='rgb_array')

        next_obs = {}
        next_obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
        next_obs['corner_rgb'] = image_obs
        next_obs['obs_gt'] = obs_gt

        terminated = info['success'] == 1

        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs_gt, info = super().reset()
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt

        image_obs = self.render(mode='rgb_array')

        obs = {}
        obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
        obs['corner_rgb'] = image_obs
        obs['obs_gt'] = obs_gt

        return obs, info


    def render(self, mode='rgb_array'):
        cam_id = mujoco.mj_name2id(self.env.model, 
                                mujoco.mjtObj.mjOBJ_CAMERA, 
                                self.camera_name)
        
        im = self.viewer.render(
            render_mode=mode,
            camera_id=cam_id
        )[::-1]
        return im
    
    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)


class MetaWorldPointcloudWrapper(gymnasium.Wrapper):
    def __init__(self, 
                 env_name: str,
                 cam_names: list,
                 img_height: int = 128,
                 img_width: int = 128,
                 max_episode_length=500,
                 num_points=256,
                 env_kwargs=None,
                 boundaries=None):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](**env_kwargs)
        env._freeze_rand_vec = False
        env.max_path_length = max_episode_length
        super().__init__(env)
        # self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self.img_height = img_height
        self.img_width = img_width
        self.boundaries = boundaries
        self.num_points = num_points

        self.cam_names = cam_names
        num_cam = len(cam_names)

        self.intrinsic_mats, self.extrinsic_mats = self.make_camera_meta_matrices()
        self.viewer = OffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        self.observation_space = gymnasium.spaces.Dict({
            # 'corner_rgb': gymnasium.spaces.Box(
            #     low=0,
            #     high=255,
            #     shape=(self.img_height, self.img_width, 3),
            #     dtype=np.uint8
            # ),
            # 'depth': gymnasium.spaces.Box(
            #     low=0,
            #     high=1,
            #     shape=(num_cam, self.img_height, self.img_width),
            #     dtype=np.float32
            # ),
            'hand_pointcloud': gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 6),
                dtype=np.float32
            ),
            # 'agent_pos': obs_gt[:4],
            'agent_pos': gymnasium.spaces.Box(
                low=np.concatenate((self.env.observation_space.low[:4], self.env.observation_space.low[18:22])).astype(np.float32),
                high=np.concatenate((self.env.observation_space.high[:4], self.env.observation_space.high[18:22])).astype(np.float32),
                # dtype=np.float32
            ),
            'hand_pos': gymnasium.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (3,),
                dtype=np.float32
            ),
            'hand_quat': gymnasium.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (4,),
                dtype=np.float32
            ),
            'hand_mat': gymnasium.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (4, 4),
                dtype=np.float32
            ),
            'hand_mat_inv': gymnasium.spaces.Box(
                low = -np.inf,
                high = np.inf,
                shape = (4, 4),
                dtype=np.float32
            ),
            
            'obs_gt': self.env.observation_space
        })

    def make_camera_meta_matrices(self):
        intrinsic_mats, extrinsic_mats = [], []
        for cam_name in self.cam_names:
            # get camera id
            cam_id = mujoco.mj_name2id(self.env.model, 
                                       mujoco.mjtObj.mjOBJ_CAMERA, 
                                       cam_name)

            # infer camera intrinsics
            fovy = math.radians(self.env.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            intrinsic_mats.append(cam_mat)

            # create extrinsic matrix
            cam_pos = self.env.model.cam_pos[cam_id]
            c2b_r = self.env.model.cam_mat0[cam_id].reshape((3, 3))
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            ext_mat = posRotMat2Mat(cam_pos, c2w_r)
            ext_mat = np.linalg.inv(ext_mat)
            extrinsic_mats.append(ext_mat)
        return np.array(intrinsic_mats), np.array(extrinsic_mats)
    
    def depth_img_2_meters(self, depth):
        extent = self.env.model.stat.extent
        near = self.env.model.vis.map.znear * extent
        far = self.env.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image
    
    def make_obs(self, obs_gt):
        ims, depths = [], []
        for cam_name in self.cam_names:
            cam_id = mujoco.mj_name2id(self.env.model, 
                                    mujoco.mjtObj.mjOBJ_CAMERA, 
                                    cam_name)
            
            im = self.viewer.render(
                render_mode='rgb_array',
                camera_id=cam_id
            )
            ims.append(im)
            depth = self.viewer.render(
                render_mode='depth_array',
                camera_id=cam_id
            )
            depth = self.depth_img_2_meters(depth)
            depths.append(depth)
            # plt.imshow(im)
            # plt.show()
            # plt.imshow(depth)
            # plt.show()
        
        ims = np.array(ims)
        depths = np.array(depths)

        # print(ims.max(), ims.min(), ims.shape)
        # print(depths.max(), depths.min(), depths.shape)
        # breakpoint()

        pcd, pcd_colors = aggr_point_cloud_from_data(
            ims[..., ::-1], 
            depths, 
            self.intrinsic_mats, 
            self.extrinsic_mats, 
            downsample=True, 
            out_o3d=False,
            max_depth=3,
            boundaries=self.boundaries,
        )#, boundaries=boundaries)
        pcd = torch.tensor(pcd, device='cuda')
        pcd_colors = torch.tensor(pcd_colors, device='cuda')

        num_points = torch.tensor([self.num_points]).cuda()
        # remember to only use coord to sample
        # breakpoint()
        # pcd_tensor = torch.tensor(pcd, device='cuda')
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=pcd.unsqueeze(0), K=num_points)

        # pcd_downsampled = pcd[sampled_indices].squeeze()
        # pcd_colors_downsampled = pcd_colors[sampled_indices].squeeze()
        pcd_downsampled = pcd[sampled_indices].detach().cpu().numpy().squeeze()
        pcd_colors_downsampled = pcd_colors[sampled_indices].detach().cpu().numpy().squeeze()
        # # sampled_indices = sampled_indices.detach().cpu().numpy()


        hand_data = self.env.data.body('hand')
        hand_pos = hand_data.xpos
        hand_quat = hand_data.xquat
        hand_rot_mat = quat2Mat(hand_quat)
        hand_mat = posRotMat2Mat(hand_pos, hand_rot_mat)
        hand_mat_inv = np.linalg.inv(hand_mat)
        # hand_mat_inv = torch.tensor(np.linalg.inv(hand_mat), device='cuda')


        # pcd_downsampled_1 = torch.cat((pcd_downsampled, torch.ones((pcd_downsampled.shape[0], 1), device='cuda')), dim=1)
        pcd_downsampled_1 = np.concatenate((pcd_downsampled, np.ones((pcd_downsampled.shape[0], 1))), axis=1)

        pcd_transformed = (hand_mat_inv @ pcd_downsampled_1.T).T[:, :3]
        # pcd_transformed = (hand_mat_inv @ pcd_downsampled_1.T).T[:, :3].detach().cpu().numpy()


        # point_cloud = np.concatenate((pcd_downsampled.detach().cpu().numpy(), pcd_colors_downsampled), axis=1)
        point_cloud = np.concatenate((pcd_transformed, pcd_colors_downsampled), axis=1)



        # breakpoint()
        # hand_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

        # show_point_cloud(pcd_downsampled.detach().cpu().numpy(), pcd_colors_downsampled)


        # fusion_obs = {
        #     'color': ims,
        #     'depth': depths,
        #     'pose': self.extrinsic_mats[:, :3], # (N, 3, 4)
        #     'K': self.intrinsic_mats,
        # }
        # self.fusion.update(fusion_obs)

        # # visualize mesh
        # init_grid, grid_shape = create_init_grid(self.boundaries, self.step_size)
        # init_grid = init_grid.to(device=self.device, dtype=torch.float32)

        # with torch.no_grad():
        #     out = self.fusion.batch_eval(init_grid, return_names=[])

        # # extract mesh
        # vertices, triangles = self.fusion.extract_mesh(init_grid, out, grid_shape)

        # # eval mask and feature of vertices
        # vertices_tensor = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)
        # with torch.no_grad():
        #     out = self.fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'color_tensor'])
        #     # out = self.fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask', 'color_tensor'])

        # # try:
        # dino_feats = out['dino_feats']
        # # except KeyError:
        # #     # breakpoint()
        # #     pass

        # num_points = torch.tensor([self.num_points]).cuda()
        # # remember to only use coord to sample
        # _, sampled_indices = torch3d_ops.sample_farthest_points(points=vertices_tensor.unsqueeze(0), K=num_points)
        # vertices_subset = vertices_tensor[sampled_indices].detach().cpu().numpy().squeeze()
        # features_subset = dino_feats[sampled_indices].detach().cpu().numpy().squeeze()

        # # do transform, scale, offset, and crop
        # if self.pc_transform is not None:
        #     vertices_subset[:, :3] = vertices_subset[:, :3] @ self.pc_transform.T
        # if self.pc_scale is not None:
        #     vertices_subset[:, :3] = vertices_subset[:, :3] * self.pc_scale
        # if self.pc_offset is not None:    
        #     vertices_subset[:, :3] = vertices_subset[:, :3] + self.pc_offset

        # point_cloud = np.concatenate([vertices_subset, features_subset], axis=1)
        
        obs_dict = {
            # 'corner_rgb': ims[0],
            'image': ims,
            'depth': depths,
            'hand_pointcloud': point_cloud,
            'agent_pos': np.concatenate((obs_gt[:4],obs_gt[18:22])),
            'obs_gt': obs_gt,
            'hand_pos': hand_pos,
            'hand_quat': hand_quat,
            'hand_mat': hand_mat,
            'hand_mat_inv': hand_mat_inv
            # 'hand_rot_mat': hand_rot_mat,
            # 'hand_mat_inv': hand_mat_inv
        }
        return obs_dict

    # def step(self, action):
    #     obs_gt, reward, terminated, truncated, info = super().step(action)
    #     obs_gt = obs_gt.astype(np.float32)
    #     info['obs_gt'] = obs_gt


    #     image_obs = self.render(mode='rgb_array')

    #     next_obs = {}
    #     next_obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
    #     next_obs['corner_rgb'] = image_obs
    #     next_obs['obs_gt'] = obs_gt

    #     terminated = info['success'] == 1

    #     return next_obs, reward, terminated, truncated, info
    
    def step(self, action):
        # Note: the obs is split up as follows:
        # obs[0:d_obs//2] current obs
        #    obs[0:3] 3d pos
        #    obs[3] gripper distance
        #    obs[4:d_obs//2] is [pos, quat] stacked for all objects
        # obs[d_obs//2:] prev obs
        obs_gt, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self.make_obs(obs_gt)

        terminated = info['success'] == 1
        # TODO: if we do RL for some reason this distinction is useful
        # done = terminated or truncated or info['success']
        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_gt, info = self.env.reset(seed=seed)
        obs_dict = self.make_obs(obs_gt)
        return obs_dict, info

    # def reset(self, seed=None, options=None):
    #     obs_gt, info = super().reset()
    #     obs_gt = obs_gt.astype(np.float32)
    #     info['obs_gt'] = obs_gt

    #     image_obs = self.render(mode='rgb_array')

    #     obs = {}
    #     obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
    #     obs['corner_rgb'] = image_obs
    #     obs['obs_gt'] = obs_gt

    #     return obs, info


    def render(self, mode='rgb_array'):
        # return np.zeros((3, 100, 100), dtype=np.uint8)
        # return self.env.render()
        cam_id = mujoco.mj_name2id(self.env.model, 
                                mujoco.mjtObj.mjOBJ_CAMERA, 
                                'corner2')
        
        im = self.viewer.render(
            render_mode=mode,
            camera_id=cam_id
        )[::-1]
        return im

    # def render(self, mode='rgb_array'):
    #     cam_id = mujoco.mj_name2id(self.env.model, 
    #                             mujoco.mjtObj.mjOBJ_CAMERA, 
    #                             self.camera_name)
        
    #     im = self.viewer.render(
    #         render_mode=mode,
    #         camera_id=cam_id
    #     )[::-1]
    #     return im
    
    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.viewer.close()


def get_env_names(benchmark=None, mode=None):
    if benchmark is None:
        return list(_env_names)
    
    if type(benchmark) is str:
        return classes[benchmark][mode]
    else:
        env_names = list(benchmark.train_classes \
            if mode == 'train' else benchmark.test_classes)
        env_names.sort()
        return env_names
    
def get_tasks(benchmark, mode):
    if benchmark is None:
        return []
    return benchmark.train_tasks if mode == 'train' else benchmark.test_tasks



def build_dataset(data_prefix, 
                  suite_name, 
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  obs_modality,
                  obs_seq_len=1, 
                  load_obs=True,
                  do_fewshot_embedding_hack=False,
                  ):
    # task_cfg = cfg.task
    task_names = get_env_names(benchmark_name, mode)
    n_tasks = len(task_names)
    # loaded_datasets = []
    datasets = []
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for task_name in task_names:
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset = get_task_dataset(
            dataset_path=os.path.join(
                data_prefix, 
                suite_name,
                benchmark_name,
                mode,
                f"{task_name}.hdf5"
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            load_obs=load_obs,
            frame_stack=frame_stack
        )
        # loaded_datasets.append(task_i_dataset)
        task_id = get_index(task_name)
        if do_fewshot_embedding_hack:
            task_id = get_hacked_task_id(task_id)
        datasets.append(SequenceVLDataset(task_i_dataset, task_id))
    n_demos = [dataset.n_demos for dataset in datasets]
    n_sequences = [dataset.total_num_sequences for dataset in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MetaWorld")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    return concat_dataset


def get_task_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    few_demos=None,
    load_obs=True,
):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
    )
    return dataset


class SequenceVLDataset(Dataset):
    # Note: task_id should be a string
    def __init__(self, sequence_dataset, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_id"] = self.task_id
        return return_dict


class ML45PRISEBenchmark(object):
    def __init__(self):
        benchmark = metaworld.ML45()
        all_classes = dict(benchmark.train_classes)
        all_classes.update(benchmark.test_classes)
        self.train_classes = {name: all_classes[name] for name in classes['ML45_PRISE']['train']}
        self.test_classes = {name: all_classes[name] for name in classes['ML45_PRISE']['test']}

        self.train_tasks = []
        self.test_tasks = []
        for task in benchmark.train_tasks + benchmark.test_tasks:
            if task.env_name in classes['ML45_PRISE']['train']:
                self.train_tasks.append(task)
            else:
                self.test_tasks.append(task)
