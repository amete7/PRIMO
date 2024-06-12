import numpy as np

import quest.utils.metaworld_utils as mu
import wandb
from tqdm import tqdm


class MetaWorldRunner():
    def __init__(self,
                 env_factory,
                 benchmark,
                 mode, # train or test
                 rollouts_per_env,
                 fps=10
                 ):
        self.env_factory = env_factory
        self.benchmark = benchmark
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.fps = fps
        

    def run(self, policy, log_video=False, do_tqdm=False):
        # print
        env_names = mu.get_env_names(self.benchmark, self.mode)
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(env_name, policy)
            for success, total_reward, episode in rollouts:
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)


                env_video.extend(episode['corner_rgb'])
            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)

            if log_video:
                video_hwc = np.array(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)
            
        # output['rollout'] = {}
        output = {}
        output['rollout'] = {
            'overall_success_rate': np.mean(successes),
            'overall_average_reward': np.mean(rewards),
            'environments_solved': int(np.sum(per_env_any_success)),
        }
        output['rollout_success_rate'] = {}
        for env_name in env_names:
            output['rollout_success_rate'][env_name] = per_env_success_rates[env_name]
            # This metric isn't that useful
            # output[f'rollout_detail/average_reward_{env_name}'] = per_env_rewards[env_name]
        if len(videos) > 0:
            output['rollout_videos'] = {}
        for env_name in videos:

            output['rollout_videos'][env_name] = videos[env_name]
        
        return output


    def run_policy_in_env(self, env_name, policy):
        env = self.env_factory(env_name=env_name)
        env_names = mu.get_env_names(self.benchmark, self.mode)
        tasks = mu.get_tasks(self.benchmark, self.mode)
        
        env_tasks = [task for task in tasks if task.env_name == env_name]
        # task_idx = env_names.index(env_name)
        count = 0
        while count < self.rollouts_per_env:
            task = env_tasks[count % len(env_tasks)]
            env.set_task(task)

            success, total_reward, episode = self.run_episode(env, 
                                                              env_name, 
                                                              policy)
            count += 1
            yield success, total_reward, episode


    def run_episode(self, env, env_name, policy):
        obs, _ = env.reset()
        if hasattr(policy, 'get_action'):
            policy.reset()
            policy = lambda obs, task_id: policy.get_action(obs, task_id)
        
        done, success, total_reward = False, False, 0

        episode = {key: [value] for key, value in obs.items()}
        episode['actions'] = []

        while not done:
            action = policy(obs, mu.get_index(env_name))
            # action = env.action_space.sample()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value)
            episode['actions'].append(action)
            if int(info["success"]) == 1:
                success = True

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode
    