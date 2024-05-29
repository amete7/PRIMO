import os
import sys
import json
import pprint
import time
from pathlib import Path
import hydra
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from utils.metaworld_utils import get_dataset, SequenceVLDataset
from utils.utils import create_experiment_dir, map_tensor_to_device, torch_save_model, get_task_names
# from primo.stage1 import SkillVAE_Model
# from primo.vqbet_vae import VQVAE_Model


def backprop(data, model, optimizer, cfg):
    data = map_tensor_to_device(data, cfg.device)
    optimizer.zero_grad()
    loss, info = model.compute_loss(data)
    loss.backward()
    if cfg.train.grad_clip is not None:
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train.grad_clip
        )
    optimizer.step()
    info.update({"grad_norm": grad_norm.item()})
    return loss.item(), info

def log_wandb(loss, info, step):
    info.update({"loss": loss})
    wandb.log(info, step=step)


@hydra.main(config_path="config", config_name="pretrain", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    pprint.pprint(cfg)
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)

    task_names = get_task_names(cfg.benchmark_name, cfg.sub_benchmark_name)
    n_tasks = len(task_names)
    print(task_names)
    loaded_datasets = []
    for i in range(n_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.data.data_dir, f"{cfg.sub_benchmark_name}/{task_names[i]}.hdf5"
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                obs_seq_len=cfg.data.obs_seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i}"
            )
            print(f"[error] {e}")
        print(f"loaded task {i}:{task_names[i]} dataset")
        loaded_datasets.append(task_i_dataset)
    print(shape_meta,'shape_meta')
    task_ids = list(range(n_tasks))
    datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(loaded_datasets, task_ids)
        ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MetaWorld")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # prepare experiment and update the config
    cfg.pretrain_model_path = ""
    create_experiment_dir(cfg)
    print(cfg.experiment_name)
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.experiment_name, save_code=True)

    # create model
    model = eval(cfg.policy.policy_type)(cfg)
    model.to(device)
    model.train()

    # start training
    optimizer = eval(cfg.train.optimizer.name)(
        model.parameters(), **cfg.train.optimizer.kwargs
    )
    scheduler = eval(cfg.train.scheduler.name)(
        optimizer, 
        T_max=cfg.train.n_epochs,
        **cfg.train.scheduler.kwargs
    )
    model_checkpoint_name = os.path.join(
            cfg.experiment_dir, f"multitask_model.pth"
        )
    concat_dataset = ConcatDataset(datasets)
    train_dataloader = DataLoader(
            concat_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            multiprocessing_context="fork",
        )
    steps = 0
    for epoch in tqdm(range(0, cfg.train.n_epochs + 1)):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        for (idx, data) in tqdm(enumerate(train_dataloader)):
            loss, info = backprop(data, model, optimizer, cfg)
            training_loss += loss
            if cfg.use_wandb:
                log_wandb(loss, info, steps)
            steps += 1
        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
        )
        if epoch % cfg.train.save_interval == 0:
            model_checkpoint_name_ep = os.path.join(
                    cfg.experiment_dir, f"multitask_model_ep{epoch}.pth"
                )
            torch_save_model(model, optimizer, scheduler, model_checkpoint_name_ep, cfg)
        scheduler.step()
    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()