import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import quest.utils.utils as utils
from pyinstrument import Profiler

OmegaConf.register_new_resolver("ppo", eval, replace=True)

from quest.utils.metaworld_vec_utils import VecEnvWrapper


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    
    # create model
    agent = instantiate(cfg.algo.policy)

    start_epoch, steps, wandb_id = 0, 0, None

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    if cfg.pretrained_path is not None:
        agent.load(cfg.pretrained_path)
        print(f"loaded pretrained model from {cfg.pretrained_path}")
    else:
        print("initialized model from scratch")
    
    if train_cfg.resume:
        checkpoint_path = utils.get_latest_checkpoint(experiment_dir)
    else: 
        checkpoint_path = cfg.checkpoint_path

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        wandb_id, start_epoch = agent.load_ckpt(checkpoint_path)

    # TODO: shift to env runner
    env = VecEnvWrapper(cfg.algo.num_envs,
                        cfg.algo.env_name,
                        cfg.algo.img_height,
                        cfg.algo.img_width,
                        cfg.algo.max_episode_length,
                        seed)
    
    print(experiment_dir)
    print(experiment_name)

    # initialize the memory and supervised pretrained policy
    agent.init(env.observation_space)

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_id,
        **cfg.logging
    )

    for iteration in range(start_epoch, cfg.algo.num_iterations):
        agent.set_eval()
        for iteration in range(cfg.algo.num_iterations):
            ep_rewards = 0
            agent.reset()
            obs, info = env.reset()
            with torch.no_grad():
                for step in range(cfg.algo.num_steps):
                    env_actions, indices, log_prob, values = agent.act(obs)
                    next_obs, rewards, terminated, truncated, info = env.step(env_actions)
                    agent.record_transition(obs, indices, rewards, values, next_obs, terminated, truncated)
                    obs = next_obs
                    ep_rewards += rewards.mean().item()

            agent.post_interaction(iteration)
            wandb.log({"episode_reward": ep_rewards}, step=iteration)
    
            if iteration % train_cfg.save_interval == 0 and iteration > 0:
                if iteration == train_cfg.n_epochs:
                    model_checkpoint_name_ep = os.path.join(
                            experiment_dir, f"multitask_model_final.pth"
                        )
                elif cfg.training.save_all_checkpoints:
                    model_checkpoint_name_ep = os.path.join(
                            experiment_dir, f"multitask_model_epoch_{iteration:04d}.pth"
                        )
                else:
                    model_checkpoint_name_ep = os.path.join(
                            experiment_dir, f"multitask_model.pth"
                        )
                agent.save(model_checkpoint_name_ep, iteration, wandb_id, experiment_dir, experiment_name)
    
    # TODO: add evaluation code
    print("[info] finished training\n")
    wandb.finish()

if __name__ == "__main__":
    main()