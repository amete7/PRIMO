import metaworld
import numpy as np
import h5py
from tqdm import tqdm
import json

import quest.utils.metaworld_utils as mu
import os

def main():
    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks
    seed = 42
    demos_per_env = 100
    max_steps = 100000
    # data_dir = '/storage/coda1/p-agarg35/0/shared/quest/data/metaworld/ML45'
    # os.makedirs(data_dir)
    success_rates = {}
    print('starting loop')

    # breakpoint()

    for name, _ in ml45.train_classes.items():
        env = mu.MetaWorldWrapper(name,max_episode_length=10)
        breakpoint()
        policy = mu.get_env_expert(name)

        factor = demos_per_env // 50 if demos_per_env >= 50 else 1
        tasks = [task for task in ml45.train_tasks if task.env_name == name]*factor
        completed = collect_demos(tasks, 
                                         env, 
                                         policy, 
                                         max_steps, 
                                         seed, 
                                        #  f'{data_dir}/{name}.hdf5',
                                         name)
        print(name, completed/len(tasks))
        success_rates.update({name: completed/len(tasks)})
    
    with open(os.path.join(data_dir, 'success_rates.json'), 'w') as f:
        json.dump(success_rates, f)

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


def collect_demos(tasks, env, policy, max_steps, seed, env_name):    
    completed = 0
    # init_hdf5(file_path, env_name)
    for i, task in enumerate(tqdm(tasks)):
        demo, success, total_reward = run_episode(env, task, policy, max_steps, seed)
        if success:
            completed += 1
        # dump_demo(demo, file_path, i)
        del demo
    return completed

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
        action = policy.get_action(obs['obs_gt'])
        action = np.random.normal(action, 0.1 * action_space_ptp)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, termn, trunc, info = env.step(action.copy())
        acts.append(action)
        print(count, termn, trunc)
        done = trunc or termn
        obs = next_obs
        total_reward += reward
        if int(info["success"]) == 1:
            success = True
    return {
        'robot_states': np.array(propris),
        'corner_rgb': np.array(cameras),
        'actions': np.array(acts)
    }, success, total_reward

def get_proprio(obs):
    return obs['obs_gt']

if __name__ == '__main__':
    main()