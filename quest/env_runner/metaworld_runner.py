import numpy as np

import quest.utils.metaworld_utils as mu
import wandb


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
        

    def run(self, policy, log_video=False):
        env_names = mu.get_env_names(self.benchmark, self.mode)

        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in env_names:

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            for success, total_reward, episode in self.run_policy_in_env(env_name, 
                                                                 policy,
                                                                 env_tasks,
                                                                 task_idx):
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
                videos[env_name] = wandb.Video(env_video, fps=self.fps)
            
        output = {
            'rollout/overall_success_rate': np.mean(successes),
            'rollout/overall_average_reward': np.mean(rewards),
            'rollout/environments_solved': np.sum(per_env_any_success),
        }
        for env_name in env_names:
            output[f'rollout_detail/success_rate_{env_name}'] = per_env_success_rates[env_name]
            output[f'rollout_detail/average_reward_{env_name}'] = per_env_rewards[env_name]
        for env_name in videos:
            output[f'rollout_videos/{env_name}'] = videos[env_name]
        
        return output




    def run_policy_in_env(self, env_name, policy):
        env = self.env_factory(env_name=env_name)
        env_names = mu.get_env_names(self.benchmark, self.mode)
        tasks = mu.get_tasks(self.benchmark, self.mode)
        
        env_tasks = [task for task in tasks if task.env_name == env_name]
        task_idx = env_names.index(env_name)
        count = 0
        while count < self.rollouts_per_env:
            task = env_tasks[count % len(tasks)]
            env.set_task(task)

            success, total_reward, episode = self.run_episode(env, 
                                                              task_idx, 
                                                              policy)
            count += 1
            yield success, total_reward, episode


    def run_episode(self, env, task_idx, policy):
        obs, _ = env.reset()
        done, success, total_reward = False, False, 0

        episode = {key: [value] for key, value in obs.items()}
        episode['actions'] = []

        while not done:
            action = policy(obs, task_idx)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

            breakpoint()

            for key, value in obs.items():
                episode[key].append(value)
            episode['actions'].append(action)
            if int(info["success"]) == 1:
                success = True
        return success, total_reward, episode
    