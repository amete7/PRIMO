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



@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    dataset = instantiate(cfg.task.dataset)
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

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

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

    if train_cfg.do_profile:
        profiler = Profiler()
    for epoch in tqdm(range(start_epoch, train_cfg.n_epochs + 1), position=0, disable=not train_cfg.use_tqdm):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        if train_cfg.do_profile:
            profiler.start()
        for idx, data in enumerate(tqdm(train_dataloader, position=1, disable=not train_cfg.use_tqdm)):
            data = utils.map_tensor_to_device(data, device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_cfg.use_amp):
                loss, info = model.compute_loss(data)
        
            scaler.scale(loss).backward()
            
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            if train_cfg.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip
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

        if train_cfg.do_profile:
            profiler.stop()
            profiler.print()

        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
        )
        if epoch % train_cfg.save_interval == 0:
            model_checkpoint_name_ep = os.path.join(
                    experiment_dir, f"multitask_model.pth"
                )
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
    print("[info] finished learning\n")
    wandb.finish()

if __name__ == "__main__":
    main()