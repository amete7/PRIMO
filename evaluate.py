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

OmegaConf.register_new_resolver("eval", eval, replace=True)



@hydra.main(config_path="config", config_name='evaluate', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    
    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # start training
    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir)

    state_dict = utils.load_state(cfg.checkpoint_path)
    model.load_state_dict(state_dict['model'])

    env_runner = instantiate(cfg.task.env_runner)
    
    print(experiment_dir)
    print(experiment_name)

    policy = lambda obs, task_id: model.get_action(obs, task_id)
    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    rollout_results = env_runner.run(policy, log_video=True, do_tqdm=train_cfg.use_tqdm)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()
    print(
        f"[info]     success rate: {rollout_results['rollout/overall_success_rate']:1.3f} \
            | environments solved: {rollout_results['rollout/environments_solved']}")


if __name__ == "__main__":
    main()