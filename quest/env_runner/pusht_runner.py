import numpy as np

import quest.utils.pusht_utils as pu
import wandb
from tqdm import tqdm


class PushTRunner():
    def __init__(self,
                 env_factory,
                 mode, # train (init state in-dist) or test (ood init state)
                 rollouts_per_env,
                 seed,
                 fps=10,
                 max_episode_length=200,
                 ):
        self.env_factory = env_factory
        self.mode = mode
        self.rollouts_per_env = rollouts_per_env
        self.seed = seed
        self.fps = fps
        self.max_episode_length = max_episode_length
        

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None):
        # print
        env_names = ['default']
        successes, per_env_any_success, rewards = [], [], []
        per_env_success_rates, per_env_rewards = {}, {}
        videos = {}
        for env_name in tqdm(env_names, disable=not do_tqdm):

            any_success = False
            env_succs, env_rews, env_video = [], [], []
            rollouts = self.run_policy_in_env(policy)
            for i, (success, total_reward, episode) in enumerate(rollouts):
                any_success = any_success or success
                successes.append(success)
                env_succs.append(success)
                env_rews.append(total_reward)
                rewards.append(total_reward)

                if i < n_video:
                    if save_video_fn is not None:
                        video_hwc = np.array(episode['corner_rgb'])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, i)
                    else:
                        env_video.extend(episode['image'])
                    
            per_env_success_rates[env_name] = np.mean(env_succs)
            per_env_rewards[env_name] = np.mean(env_rews)
            per_env_any_success.append(any_success)

            if len(env_video) > 0:
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


    def run_policy_in_env(self, policy):
        env = self.env_factory()
        if self.mode=='train':
            seed = self.seed if self.seed<200 else np.random.randint(0,200)
        else:
            seed = self.seed+201
        env.seed(seed)
        count = 0
        while count < self.rollouts_per_env:
            success, total_reward, episode = self.run_episode(env, 
                                                              policy)
            count += 1
            yield success, total_reward, episode


    def run_episode(self, env, policy):
        obs, _ = env.reset()
        # breakpoint()
        if hasattr(policy, 'get_action'):
            policy.reset()
            policy_object = policy
            policy = lambda obs, task_id, task_emb, batchify: policy_object.get_action(obs, task_id, task_emb, batchify)
        
        done, success, total_reward = False, False, 0

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode['actions'] = []

        task_id = 0
        task_emb = None
        batchify = True

        steps = 0
        while steps < self.max_episode_length:
            action = policy(obs, task_id, task_emb, batchify)
            # action = env.action_space.sample()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            total_reward += reward
            obs = next_obs

            # breakpoint()
            for key, value in obs.items():
                episode[key].append(value[-1])
            episode['actions'].append(action)
            
            if done:
                success = True
                break
            steps+=1

        episode = {key: np.array(value) for key, value in episode.items()}
        return success, total_reward, episode
    