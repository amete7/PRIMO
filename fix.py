import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import quest.utils.utils as utils
from pyinstrument import Profiler
from quest.utils.logger import Logger

OmegaConf.register_new_resolver("eval", eval, replace=True)



@hydra.main(config_path="config", version_base=None)
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
    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg, allow_overlap=True)
    os.makedirs(experiment_dir, exist_ok=True)

    start_epoch, steps, wandb_id = 0, 0, None
    checkpoint_path = experiment_dir
    
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f'loading from checkpoint {checkpoint_path}')
        state_dict = utils.load_state(checkpoint_path)
        loaded_state_dict = state_dict['model']
        
        # TODO: This is a hack to allow loading state dicts with some mismatched parameters
        # might want to remove
        utils.soft_load_state_dict(model, loaded_state_dict)

        # resuming training since we are loading a checkpoint training the same stage
        if cfg.stage == state_dict['stage']:
            print('loading from checkpoint')
            for optimizer, opt_state_dict in zip(optimizers, state_dict['optimizers']):
                optimizer.load_state_dict(opt_state_dict)
            for scheduler, sch_state_dict in zip(schedulers, state_dict['schedulers']):
                scheduler.load_state_dict(sch_state_dict)
            scaler.load_state_dict(state_dict['scaler'])
            start_epoch = state_dict['epoch']
            steps = state_dict['steps']
            wandb_id = state_dict['wandb_id']
        # elif train_cfg.auto_continue:
        #     wandb_id = state_dict['wandb_id']
    else:
        print('starting from scratch')

    print(experiment_dir)
    print(experiment_name)

    utils.save_state({
        'model': model,
        'optimizers': optimizers,
        'schedulers': schedulers,
        'scaler': scaler,
        'epoch': start_epoch,
        'stage': cfg.stage,
        'steps': steps,
        'wandb_id': wandb_id,
        'experiment_dir': experiment_dir,
        'experiment_name': experiment_name,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, checkpoint_path)

if __name__ == "__main__":
    main()