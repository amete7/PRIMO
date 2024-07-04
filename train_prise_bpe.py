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



@hydra.main(config_path="config", config_name='train_prise_bpe', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    dataset = instantiate(cfg.task.dataset)
    
    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # TODO: once we verify that this script works update the following line to allow it to overwrite saved checkpoints to add BPE
    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    if train_cfg.auto_continue:
        checkpoint_path = os.path.join(experiment_dir, f'stage_0')
    elif train_cfg.resume: 
        assert False, 'resume training is not supported for BPE'
    else: 
        checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f'loading from checkpoint {checkpoint_path}')
        state_dict = utils.load_state(checkpoint_path)
        loaded_state_dict = state_dict['model']
        
        utils.soft_load_state_dict(model, loaded_state_dict)
        # model.load_state_dict(loaded_state_dict)

        # elif train_cfg.auto_continue:
        #     wandb_id = state_dict['wandb_id']
    else:
        pass # TODO: add assert back
        # assert False, 'checkpoint required'

    
    print(experiment_dir)
    print(experiment_name)

    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    model.train_bpe(dataset, use_tqdm=train_cfg.use_tqdm)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()

    utils.save_state({
        'model': model,
        'optimizers': state_dict['optimizers'],
        'schedulers': state_dict['schedulers'],
        'scaler': state_dict['scaler'],
        'epoch': state_dict['epoch'],
        'stage': cfg.stage,
        'steps': state_dict['steps'],
        'wandb_id': state_dict['wandb_id'],
        'experiment_dir': experiment_dir,
        'experiment_name': experiment_name,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, os.path.join(experiment_dir, 'multitask_model_bpe.pth'))

    # for epoch in range(start_epoch, train_cfg.n_epochs + 1):
    #     t0 = time.time()
    #     model.train()
    #     training_loss = 0.0
    #     if train_cfg.do_profile:
    #     for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
    #         data = utils.map_tensor_to_device(data, device)
            
    #         with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=train_cfg.use_amp):
    #             loss, info = model.compute_loss(data)
        
    #         scaler.scale(loss).backward()
            
    #         for optimizer in optimizers:
    #             scaler.unscale_(optimizer)
    #         if train_cfg.grad_clip is not None:
    #             grad_norm = nn.utils.clip_grad_norm_(
    #                 model.parameters(), train_cfg.grad_clip
    #             )

    #         # optimizer.step()
    #         for optimizer in optimizers:
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad()

    #         info.update({
    #             "grad_norm": grad_norm.item(),
    #             'epoch': epoch
    #         })
    #         info = {cfg.logging_folder: info}
    #         training_loss += loss
    #         # wandb.log(info, step=steps)
    #         steps += 1
    #         logger.update(info, steps)


    #     training_loss /= len(train_dataloader)
    #     t1 = time.time()
    #     print(
    #         f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {(t1-t0)/60:4.2f}"
    #     )

    #     # if cfg.rollout.enabled and epoch % cfg.rollout.interval == 0:
    #     if cfg.rollout.enabled and epoch > 0 and epoch % cfg.rollout.interval == 0:
    #         # policy = lambda obs, task_id: model.get_action(obs, task_id)
    #         rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=train_cfg.use_tqdm)
    #         print(
    #             f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
    #                 | environments solved: {rollout_results['rollout']['environments_solved']}")
    #         logger.log(rollout_results, step=steps)
        
    #     if epoch % train_cfg.save_interval == 0 and epoch > 0:
    #         if epoch == train_cfg.n_epochs:
    #             model_checkpoint_name_ep = os.path.join(
    #                     experiment_dir, f"multitask_model_final.pth"
    #                 )
    #         elif cfg.training.save_all_checkpoints:
    #             model_checkpoint_name_ep = os.path.join(
    #                     experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth"
    #                 )
    #         else:
    #             model_checkpoint_name_ep = os.path.join(
    #                     experiment_dir, f"multitask_model.pth"
    #                 )
    #         utils.save_state({
    #             'model': model,
    #             'optimizers': optimizers,
    #             'schedulers': schedulers,
    #             'scaler': scaler,
    #             'epoch': epoch,
    #             'stage': cfg.stage,
    #             'steps': steps,
    #             'wandb_id': wandb.run.id,
    #             'experiment_dir': experiment_dir,
    #             'experiment_name': experiment_name,
    #             'config': OmegaConf.to_container(cfg, resolve=True)
    #         }, model_checkpoint_name_ep)
    #     [scheduler.step() for scheduler in schedulers]
    # print("[info] finished learning\n")
    # wandb.finish()

if __name__ == "__main__":
    main()