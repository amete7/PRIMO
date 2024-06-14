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
        obs_gt = obs['obs_gt'].squeeze()
        return env_experts[_env_names[task_id]].get_action(obs_gt)
    
    return expert

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
                  obs_seq_len, 
                  obs_modality,
                  load_obs=True
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
            load_obs=load_obs
        )
        # loaded_datasets.append(task_i_dataset)
        datasets.append(SequenceVLDataset(task_i_dataset, get_index(task_name)))
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
    *args,
    **kwargs
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
        # obs_seq_length=obs_seq_len,
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
