import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path

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

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    start_epoch, steps, wandb_id = 0, 0, None
    if train_cfg.auto_continue:
        checkpoint_path = os.path.join(experiment_dir, os.path.pardir, 'stage_0/multitask_model_final.pth')
    if train_cfg.resume: 
        # # TODO once we are to the next generation of models with updated saving we can use this simpler logic
        # # onlyfiles = [f for f in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, f))]
        # # onlyfiles.sort()
        # # checkpoint_path = onlyfiles[-1]

        # latest = 0
        # if os.path.exists(experiment_dir):
        #     for path in Path(experiment_dir).glob("multitask_model_epoch_*"):
        #         try:
        #             folder_id = int(str(path).split("_")[-1][:-4])
        #             if folder_id > latest:
        #                 latest = folder_id
        #         except BaseException:
        #             pass
        # checkpoint_path = f'{experiment_dir}/multitask_model_epoch_{latest}.pth'
        checkpoint_path = utils.get_latest_checkpoint(experiment_dir)
    else: 
        checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        state_dict = utils.load_state(checkpoint_path)
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
        # elif train_cfg.auto_continue:
        #     wandb_id = state_dict['wandb_id']

    if cfg.rollout.enabled:
        env_runner = instantiate(cfg.task.env_runner)
    
    print(experiment_dir)
    print(experiment_name)

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_id,
        **cfg.logging
    )

    logger = Logger(train_cfg.log_interval)
    if train_cfg.do_profile:
        profiler = Profiler()
    for epoch in range(start_epoch, train_cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        if train_cfg.do_profile:
            profiler.start()
        for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
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
            # wandb.log(info, step=steps)
            steps += 1
            logger.update(info, steps, epoch)

        if train_cfg.do_profile:
            profiler.stop()
            profiler.print()

        training_loss /= len(train_dataloader)
        t1 = time.time()
        print(
            f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
        )

        # if cfg.rollout.enabled and epoch % cfg.rollout.interval == 0:
        if cfg.rollout.enabled and epoch > 0 and epoch % cfg.rollout.interval == 0:
            policy = lambda obs, task_id: model.get_action(obs, task_id)
            rollout_results = env_runner.run(policy, log_video=True, do_tqdm=train_cfg.use_tqdm)
            print(
                f"[info]     success rate: {rollout_results['rollout/overall_success_rate']:1.3f} \
                    | environments solved: {rollout_results['rollout/environments_solved']}")
            wandb.log(rollout_results, step=steps)
        
        if epoch % train_cfg.save_interval == 0 and epoch > 0:
            if epoch == train_cfg.n_epochs:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_final.pth"
                    )
            elif cfg.training.save_all_checkpoints:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth"
                    )
            else:
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