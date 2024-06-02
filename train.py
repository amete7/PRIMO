import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from quest.utils.metaworld_utils import get_dataset, SequenceVLDataset
import quest.utils.utils as utils
# create_experiment_dir, map_tensor_to_device, get_task_names, save_state

OmegaConf.register_new_resolver("eval", eval, replace=True)


def build_dataset(task_cfg):
    task_names = utils.get_task_names(task_cfg.benchmark_name, task_cfg.sub_benchmark_name)
    n_tasks = len(task_names)
    loaded_datasets = []
    for i in range(n_tasks):
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


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)

    dataset = build_dataset(cfg.task)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    
    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # start training
    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)

    start_epoch, steps, wandb_id, experiment_dir, experiment_name = 0, 0, None, None, None
    if cfg.checkpoint_path is not None:
        state_dict = utils.load_state(cfg.checkpoint_path)
        model.load_state_dict(state_dict['model'])

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
            experiment_dir = state_dict['experiment_dir']
            experiment_name = state_dict['experiment_name']

    if experiment_dir is None:
        experiment_dir, experiment_name = utils.create_experiment_dir(cfg)
    
    print(experiment_dir)
    print(experiment_name)

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_id,
        **cfg.logging
    )

    # from pyinstrument import Profiler
    steps = 0
    for epoch in tqdm(range(start_epoch, cfg.training.n_epochs + 1), position=0, disable=not cfg.training.use_tqdm):
        # profiler = Profiler()
        # profiler.start()
        t0 = time.time()
        model.train()
        training_loss = 0.0
        for idx, data in enumerate(tqdm(train_dataloader, position=1, disable=not cfg.training.use_tqdm)):
            # loss, info = backprop(data, model, optimizer, cfg.training.grad_clip, device)
            data = utils.map_tensor_to_device(data, device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.training.use_amp):
                loss, info = model.compute_autoencoder_loss(data)
            
            scaler.scale(loss).backward()
            
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            if cfg.training.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip
                )

            # optimizer.step()
            for optimizer in optimizers:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            info.update({"grad_norm": grad_norm.item()})
            training_loss += loss
            wandb.log(info, step=steps)
            steps += 1

        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
        )
        if epoch % cfg.training.save_interval == 0:
            model_checkpoint_name_ep = os.path.join(
                    experiment_dir, f"multitask_model.pth"
                )
            # torch_save_model(model, optimizer, scheduler, model_checkpoint_name_ep, cfg)
            utils.save_state({
                'model': model,
                'optimizers': optimizers,
                'schedulers': schedulers,
                'scaler': scaler,
                'epoch': epoch,
                'stage': cfg.stage,
                'steps': steps,
                'wandb_id': wandb.run.id,
                'experiment_dir': experiment_dir,
                'experiment_name': experiment_name,
            }, model_checkpoint_name_ep)
        [scheduler.step() for scheduler in schedulers]
        # profiler.stop()
        # profiler.print()
    print("[info] finished learning\n")
    wandb.finish()

if __name__ == "__main__":
    main()