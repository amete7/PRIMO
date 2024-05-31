import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from quest.utils.metaworld_utils import get_dataset, SequenceVLDataset
from quest.utils.utils import create_experiment_dir, map_tensor_to_device, torch_save_model, get_task_names

OmegaConf.register_new_resolver("eval", eval, replace=True)

def backprop(data, model, optimizer, grad_clip, device):
    data = map_tensor_to_device(data, device)
    optimizer.zero_grad()
    loss, info = model.compute_loss(data)
    loss.backward()
    if grad_clip is not None:
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        )
    optimizer.step()
    info.update({"grad_norm": grad_norm.item()})
    return loss.item(), info

def log_wandb(loss, info, step):
    info.update({"loss": loss})
    wandb.log(info, step=step)

def build_dataset(task_cfg):
    task_names = get_task_names(task_cfg.benchmark_name, task_cfg.sub_benchmark_name)
    n_tasks = len(task_names)
    # print(task_names)
    loaded_datasets = []
    for i in trange(n_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                task_cfg.data_dir, f"{task_cfg.sub_benchmark_name}/{task_names[i]}.hdf5"
            ),
            obs_modality=task_cfg.obs_modality,
            initialize_obs_utils=(i == 0),
            seq_len=task_cfg.seq_len,
            obs_seq_len=task_cfg.obs_seq_len,
        )
        loaded_datasets.append(task_i_dataset)
    task_ids = list(range(n_tasks))
    datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(loaded_datasets, task_ids)
        ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MetaWorld")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    return concat_dataset


@hydra.main(config_path="config", config_name="pretrain", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)

    dataset = build_dataset(cfg.task)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    

    # prepare experiment and update the config
    # cfg.pretrain_model_path = ""
    experiment_dir, experiment_name = create_experiment_dir(cfg)
    print(experiment_name)
    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging
    )

    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # start training
    # TODO: replace the rest of the 
    optimizer = instantiate(cfg.optimizer,
                            params=model.parameters())
    scheduler = instantiate(cfg.scheduler,
                            optimizer=optimizer)
    model_checkpoint_name = os.path.join(
            experiment_dir, f"multitask_model.pth"
        )
    # breakpoint()
    # print('if you get here you deserve a medal')
    steps = 0
    for epoch in tqdm(range(0, cfg.training.n_epochs + 1), position=0):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        for (idx, data) in enumerate(tqdm(train_dataloader, position=1)):
            # loss, info = backprop(data, model, optimizer, cfg.training.grad_clip, device)
            data = map_tensor_to_device(data, device)
            
            optimizer.zero_grad()
            loss, info = model.compute_autoencoder_loss(data)
            loss.backward()
            if cfg.training.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip
                )
            optimizer.step()
            info.update({"grad_norm": grad_norm.item()})
            training_loss += loss
            log_wandb(loss, info, steps)
            steps += 1
        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
        )
        # if epoch % cfg.train.save_interval == 0:
        #     model_checkpoint_name_ep = os.path.join(
        #             experiment_dir, f"multitask_model_ep{epoch}.pth"
        #         )
        #     torch_save_model(model, optimizer, scheduler, model_checkpoint_name_ep, cfg)
        scheduler.step()
    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()