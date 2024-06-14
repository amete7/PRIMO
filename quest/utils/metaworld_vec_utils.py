import copy
from collections import OrderedDict
import random
from typing import Any, Optional
import numpy as np
import quest.utils.file_utils as FileUtils
import quest.utils.obs_utils as ObsUtils
from PIL import Image
from quest.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
import mujoco
import os
from torch.utils.data import ConcatDataset
import metaworld

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


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
            'disassemble-v2',
            'pick-place-wall-v2',
            'stick-pull-v2',
            'box-close-v2',
            'hand-insert-v2',
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
        return env_experts[task_id].get_action(obs['obs_gt'])
    
    return expert

def get_env_expert(env_name):
    return _policies[env_name]()


class MetaWorldWrapperVec(gymnasium.Wrapper):
    def __init__(self, 
                 env_name: str,
                 all_tasks: list,
                 img_height: int = 128,
                 img_width: int = 128,
                 max_episode_length=500,
                 camera_name='corner2',
                 env_kwargs=None,):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](render_mode='rgb_array')
        env._freeze_rand_vec = False
        env.max_path_length = max_episode_length
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        super().__init__(env)
        self.task_list = [task for task in all_tasks if task.env_name == env_name]

    def step(self, actions):
        rewards = 0
        assert actions.ndim == 2, "actions must have shape (horizon, action_dim)"
        for action in actions:
            obs_gt, reward, terminated, truncated, info = super().step(action)
            rewards += reward
        info['obs_gt'] = obs_gt
        terminated = info['success'] == 1
        return obs_gt, rewards, terminated, truncated, info
    
    def reset(self, seed=None, **kwargs):
        if len(self.task_list) > 0:
            task = random.choice(self.task_list)
            self.set_task(task)
        obs_gt, info = super().reset(seed=seed, **kwargs)
        info['obs_gt'] = obs_gt
        return obs_gt, info
    
    def set_task(self, task):
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

class VecEnvWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, num_envs, env_name, mode, img_height=128, img_width=128, max_episode_length=500, debug=True, camera_name='corner2', device='cuda', env_kwargs=None):
        if not debug:
            benchmark = metaworld.ML45()
            all_tasks = get_tasks(benchmark, mode)
        else:
            all_tasks = []
        env_list = [lambda: MetaWorldWrapperVec(env_name, all_tasks, img_height, img_width, max_episode_length, camera_name, env_kwargs) for _ in range(num_envs)]
        env = gymnasium.vector.AsyncVectorEnv(env_list, context='spawn')
        super().__init__(env)
        self.single_env_space = gymnasium.spaces.Dict({
            'robot_state': gymnasium.spaces.Box(
                low=np.concatenate((self.env.single_observation_space.low[:4], self.env.single_observation_space.low[18:22])),
                high=np.concatenate((self.env.single_observation_space.high[:4], self.env.single_observation_space.high[18:22])),
                dtype=np.float32
            ),
            'corner_rgb': gymnasium.spaces.Box(
                low=0,
                high=1,
                shape=(3, img_height, img_width),
                dtype=np.float32
            ),
            'obs_gt': self.env.single_observation_space
        })
        self.device = device
        self.observation_space = gymnasium.vector.utils.batch_space(self.single_env_space, num_envs)

    def reset(self, seed=None, **kwargs):
        obs_gt, info = self.env.reset(seed=seed, **kwargs)
        obs = self.get_obs(obs_gt)
        return obs, info

    def step(self, actions):
        obs_gt, rewards, terminated, truncated, info = self.env.step(actions)
        obs = self.get_obs(obs_gt)
        return obs, rewards.astype(np.float32), terminated, truncated, info

    def get_obs(self, obs_gt):
        image_obs = self.env.call('render')
        image_obs = np.array(image_obs)[:,::-1]
        image_obs = np.transpose(image_obs, (0, 3, 1, 2))
        image_obs = image_obs.astype(np.float32) / 255.0
        image_obs = np.clip(image_obs, 0, 1)
        obs = {}
        obs['robot_state'] = np.concatenate((obs_gt[:,:4],obs_gt[:,18:22]), axis=1, dtype=np.float32)
        obs['corner_rgb'] = image_obs
        obs['obs_gt'] = obs_gt.astype(np.float32)
        obs = self._observation_to_tensor(obs)
        return obs

    def _observation_to_tensor(self, observation: Any, space: Optional[gymnasium.Space] = None) -> torch.Tensor:
        """Convert the Gymnasium observation to a flat tensor

        :param observation: The Gymnasium observation to convert to a tensor
        :type observation: Any supported Gymnasium observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        observation_space = self.observation_space
        space = space if space is not None else observation_space

        if isinstance(space, gymnasium.spaces.MultiDiscrete):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, int):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Discrete):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Box):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(space, gymnasium.spaces.Dict):
            tmp = torch.cat([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], dim=-1).view(self.num_envs, -1)
            return tmp
        else:
            raise ValueError(f"Observation space type {type(space)} not supported. Please report this issue")

def get_env_names(benchmark, mode):
    if type(benchmark) is str:
        return classes[benchmark][mode]
    else:
        env_names = list(benchmark.train_classes \
            if mode == 'train' else benchmark.test_classes)
        env_names.sort()
        return env_names
    
def get_tasks(benchmark, mode):
    return benchmark.train_tasks if mode == 'train' else benchmark.test_tasks



if __name__ == '__main__':
    benchmark = 'ML45'
    mode = 'test'
    env_names = get_env_names(benchmark, mode)
    name = env_names[0]
    num_envs = 20
    env = VecEnvWrapper(num_envs, name, mode)
    print(env.action_space)
    print(env.observation_space)
    obs, info = env.reset()
    i=0
    while True:
        actions = env.action_space.sample()
        actions = np.expand_dims(actions, axis=1)
        actions = np.tile(actions, (1, 16, 1)) # action chunk of size 16
        next_obs, reward, terminated, truncated, info = env.step(actions)
        print(reward, terminated, truncated, i)
        i += 1
        if i>28: # in order to not exceed the max_episode_length
            obs, info = env.reset() # manual reset required after truncation
            i = 0

