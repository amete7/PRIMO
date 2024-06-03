import metaworld
import numpy as np
import h5py
from tqdm import tqdm
import json
import torch

import quest.utils.metaworld_utils as mu
import os
from quest.utils.utils import create_experiment_dir, map_tensor_to_device, torch_load_model, get_task_names


class MetaworldRunner():
    def __init__(self,
                 env_factory,
                 benchmark,
                 mode, # train or test

                 max_steps,
                 device,
                 ):
        self.env_factory = env_factory
        self.benchmark = benchmark
        self.mode = mode
        self.max_steps = max_steps
        self.device = device

    def run(self, policy, tasks_per_env=100):
        classes = self.benchmark.train_classes \
            if self.mode == 'train' else self.benchmark.test_classes
        tasks = self.benchmark.train_tasks \
            if self.mode == 'train' else self.benchmark.test_tasks
        
        for name in classes:
            env = self.env_factory(name=name)
            env_tasks = [task for task in tasks if task.env_name == name]
            count = 0
            
            while count < tasks_per_env:
                demo, success, total_reward = self.run_episode(env, task, policy, max_steps, seed)
                if success:
                    completed += 1
                dump_demo(demo, file_path, i)
                del demo


            completed = 0
            # for i, task in enumerate(tqdm(tasks)):
            return completed


    def run_episode(self, env, task_idx, policy, i):
        obs, _ = env.reset()
        done, success = False, False

        # frames = [env.render()]
        episode = {key: [value] for key, value in obs.items()}
        episode['actions'] = []

        while not done and not success:
            action = policy.get_action(obs, task_idx)
            action = np.squeeze(action, axis=0)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            # action = torch.tensor(action)
            next_obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value)
            episode['actions'].append(action)
            # frames.append(env.render())
            # frames.append(next_obs['corner_rgb'])
            # obs_input = get_data(next_obs, frames[-1], task_idx)
            if int(info["success"]) == 1:
                # print("Success")
                success = True
        # if video_dir is not None:
        #     imageio.mimsave(
        #                 os.path.join(video_dir, f'{task_idx}_{i}.mp4'), frames, fps=15)
        return success, episode
    
    # def get_data(self, obs, image, task_id):
    #     batch = {}
    #     batch["obs"] = {}
    #     batch["obs"]['robot_states'] = torch.tensor((np.concatenate((obs[:4],obs[18:22]))), dtype=torch.float32).unsqueeze(0)
    #     image_obs = process_frame(frame=image, channel_dim=3, scale=255.)
    #     batch["obs"]['corner_rgb'] = torch.tensor(image_obs).unsqueeze(0)
    #     batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
    #     batch = map_tensor_to_device(batch, device)
    #     return batch



def main():
    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks
    seed = 42
    demos_per_env = 100
    max_steps = 500
    data_dir = '/storage/coda1/p-agarg35/0/shared/quest/data/metaworld/ML45'
    os.makedirs(data_dir)
    success_rates = {}
    print('starting loop')
    for name, _ in ml45.train_classes.items():
        env = mu.MetaWorldWrapper(name)
        policy = mu.get_expert(name)

        factor = demos_per_env // 50 if demos_per_env >= 50 else 1
        tasks = [task for task in ml45.train_tasks if task.env_name == name]*factor
        completed = collect_demos(tasks, 
                                         env, 
                                         policy, 
                                         max_steps, 
                                         seed, 
                                         f'{data_dir}/{name}.hdf5',
                                         name)
        print(name, completed/len(tasks))
        success_rates.update({name: completed/len(tasks)})
    
    with open(os.path.join(data_dir, 'success_rates.json'), 'w') as f:
        json.dump(success_rates, f)


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