import metaworld
import numpy as np
import h5py
from tqdm import tqdm
import json

import quest.utils.metaworld_utils as mu
import os
import hydra
from hydra.utils import instantiate
import quest.utils.utils as utils
from moviepy.editor import ImageSequenceClip

@hydra.main(config_path="../config", 
            config_name='collect_data', 
            version_base=None)
def main(cfg):
    env_runner = instantiate(cfg.task.env_runner)

    data_dir = os.path.join(
                cfg.data_prefix, 
                cfg.task.benchmark_name,
                cfg.task.sub_benchmark_name,
                # f"{task_names[i]}.hdf5"
            )
    experiment_dir, _ = utils.create_experiment_dir(cfg)
    
    # os.makedirs(data_dir)
    success_rates, returns = {}, {}
    expert = mu.get_expert(cfg.task.sub_benchmark_name, cfg.task.mode)

    # TODO: this assumes a [-1, 1] action space which is true for metaworld but not universal
    def noisy_expert(obs, task_idx):
        expert_action = expert(obs, task_idx)
        action = np.random.normal(expert_action, 0.2)
        action = np.clip(action, -1, 1)
        return action

    for env_name in mu.get_env_names(cfg.task.sub_benchmark_name, cfg.task.mode):
        file_path = os.path.join(data_dir, f"{env_name}.hdf5")
        video_dir = os.path.join(experiment_dir, env_name)
        # init_hdf5(file_path, env_name)
        
        completed = total_return = 0
        rollouts = env_runner.run_policy_in_env(env_name, noisy_expert)
        for i, (success, ep_return, episode) in tqdm(enumerate(rollouts), total=cfg.rollouts_per_env):

            completed += success
            total_return += ep_return

            save_path = os.path.join(video_dir, f'trial_{i}.mp4')
            clip = ImageSequenceClip(episode['corner_rgb'], fps=24)
            clip.write_videofile(save_path, fps=24)
            # dump_demo(episode, file_path, i)
        success_rate = completed / (i + 1)
        success_rates.update({env_name: success_rate})
        print(env_name, success_rate)

    # with open(os.path.join(data_dir, 'success_rates.json'), 'w') as f:
    #     json.dump(success_rates, f)

    # for env_name, _ in ml45.train_classes.items():
    #     env = mu.MetaWorldWrapper(env_name)
    #     policy = mu.get_env_expert(env_name)

    #     factor = demos_per_env // 50 if demos_per_env >= 50 else 1
    #     tasks = [task for task in ml45.train_tasks if task.env_name == env_name]*factor
    #     completed = collect_demos(tasks, 
    #                                      env, 
    #                                      policy, 
    #                                      max_steps, 
    #                                      seed, 
    #                                      f'{data_dir}/{env_name}.hdf5',
    #                                      env_name)


    # seed = 42
    # print('starting loop')
    

def collect_demos(tasks, env, policy, max_steps, seed, file_path, env_name):    
    completed = 0
    for i, task in enumerate(tqdm(tasks)):
        demo, success, total_reward = run_episode(env, task, policy, max_steps, seed)
        if success:
            completed += 1
        del demo
    return completed

def init_hdf5(file_path, env_name):
    with h5py.File(file_path, 'a') as f:
        group_data = f.create_group('data')
        group_data.attrs['total'] = 0
        group_data.attrs['env_args'] = json.dumps({
            'env_name': env_name, 'env_type': 2, 
            'env_kwargs':{'render_mode':'rgb_array', 'camera_name':'corner2'}
            })

def dump_demo(demo, file_path, demo_i):
    with h5py.File(file_path, 'a') as f:
        group_data = f['data']
        group = group_data.create_group(f'demo_{demo_i}')

        demo_length = demo['actions'].shape[0]
        group_data.attrs['total'] = group_data.attrs['total'] + demo_length
        group.attrs['num_samples'] = demo_length
        group.create_dataset('states', data=())
        group.create_dataset('obs/robot_states', data=demo['robot_states'])
        group.create_dataset('obs/corner_rgb', data=demo['corner_rgb'])
        group.create_dataset('actions', data=demo['actions'])



def run_episode(env, task, policy, max_steps=500, seed=42):
    propris, cameras, acts = [], [], []
    env.set_task(task)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    action_space_ptp = env.action_space.high - env.action_space.low
    obs, _ = env.reset()
    done, success = False, False
    count = 0
    total_reward = 0
    while count < max_steps and not done:
        propris.append(get_proprio(obs))
        cameras.append(env.render())

        count += 1
        action = policy.get_action(obs)
        action = np.random.normal(action, 0.1 * action_space_ptp)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, trunc, termn, info = env.step(action.copy())
        acts.append(action)
        done = trunc or termn
        obs = next_obs
        total_reward += reward
        if int(info["success"]) == 1:
            success = True
            break
    return {
        'robot_states': np.array(propris),
        'corner_rgb': np.array(cameras),
        'actions': np.array(acts)
    }, success, total_reward

def get_proprio(obs):
    return np.concatenate((obs[:4],obs[18:22]))

if __name__ == '__main__':
    main()