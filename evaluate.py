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
from moviepy.editor import ImageSequenceClip
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)

{'_target_': 'quest.algos.quest.QueST', 'autoencoder': {'_target_': 'quest.algos.quest_modules.skill_vae.SkillVAE', 'action_dim': 4, 'encoder_dim': 256, 'decoder_dim': 256, 'skill_block_size': 16, 'downsample_factor': 2, 'encoder_heads': 4, 'encoder_layers': 2, 'decoder_heads': 4, 'decoder_layers': 4, 'attn_pdrop': 0.1, 'use_causal_encoder': True, 'use_causal_decoder': True, 'vq_type': 'fsq', 'fsq_level': None, 'codebook_dim': 512, 'codebook_size': 512}, 'policy_prior': {'_target_': 'quest.algos.quest_modules.skill_gpt.SkillGPT', 'action_dim': 4, 'start_token': 1000, 'offset_layers': 2, 'offset_hidden_dim': 512, 'offset_dim': 64, 'vocab_size': 1000, 'block_size': 8, 'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'attn_pdrop': 0.1, 'embd_pdrop': 0.1, 'beam_size': 5, 'temperature': 1.0, 'device': 'cuda:0'}, 'image_encoder_factory': {'_target_': 'quest.algos.utils.rgb_modules.ResnetEncoder', '_partial_': True, 'input_shape': [3, 128, 128], 'output_size': 256, 'pretrained': False, 'freeze': False, 'remove_layer_num': 4, 'no_stride': False, 'language_fusion': 'none'}, 'proprio_encoder': {'_target_': 'quest.algos.utils.mlp_proj.MLPProj', 'input_size': 8, 'output_size': 128, 'num_layers': 1}, 'obs_proj': {'_target_': 'quest.algos.utils.mlp_proj.MLPProj', 'input_size': 384, 'output_size': 384}, 'task_encoder': {'_target_': 'torch.nn.Embedding', 'num_embeddings': 50, 'embedding_dim': 384}, 'image_aug': {'_target_': 'quest.algos.utils.data_augmentation.DataAugGroup', 'aug_list': [{'_target_': 'quest.algos.utils.data_augmentation.BatchWiseImgColorJitterAug', 'input_shape': [3, 128, 128], 'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.3, 'epsilon': 0.1}, {'_target_': 'quest.algos.utils.data_augmentation.TranslationAug', 'input_shape': [3, 128, 128], 'translation': 4}]}, 'loss_fn': {'_target_': 'torch.nn.L1Loss'}, 'optimizer_factory': {'_target_': 'torch.optim.AdamW', '_partial_': True, 'lr': 0.0001, 'betas': [0.9, 0.999], 'weight_decay': 0.0001}, 'scheduler_factory': {'_target_': 'torch.optim.lr_scheduler.CosineAnnealingLR', '_partial_': True, 'eta_min': 1e-05, 'last_epoch': -1, 'T_max': 1000}, 'stage': 2, 'l1_loss_scale': 100, 'action_horizon': 2, 'shape_meta': {'action_dim': 4, 'proprio_dim': 8, 'image_shape': [3, 128, 128], 'image_inputs': ['corner_rgb']}, 'device': 'cuda:0'}

@hydra.main(config_path="config", config_name='evaluate', version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    OmegaConf.resolve(cfg)
    
    # create model

    save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True)
    os.makedirs(save_dir)

    if cfg.checkpoint_path is None:
        # Basically if you don't provide a checkpoint path it will automatically find one corresponding
        # to the experiment/variant name you provide
        checkpoint_path, _ = utils.get_experiment_dir(cfg, evaluate=False, allow_overlap=True)
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
    state_dict = utils.load_state(checkpoint_path)
    
    if 'config' in state_dict:
        print('autoloading based on saved parameters')
        model = instantiate(state_dict['config']['algo']['policy'], 
                            shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy,
                            shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()
    
    model.load_state_dict(state_dict['model'])

    env_runner = instantiate(cfg.task.env_runner)
    
    print(save_dir)

    def save_video_fn(video_chw, env_name, idx):
        video_dir = os.path.join(save_dir, 'videos', env_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, f'{idx}.mp4')
        clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
        clip.write_videofile(save_path, fps=24, verbose=False, logger=None)

    # policy = lambda obs, task_id: model.get_action(obs, task_id)
    if train_cfg.do_profile:
        profiler = Profiler()
        profiler.start()
    rollout_results = env_runner.run(model, n_video=50, do_tqdm=train_cfg.use_tqdm, save_video_fn=save_video_fn)
    if train_cfg.do_profile:
        profiler.stop()
        profiler.print()
    print(
        f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} \
            | environments solved: {rollout_results['rollout']['environments_solved']}")

    # videos = {key.split('/')[1]: value.data for key, value in rollout_results.items() if 'rollout_videos' in key}
        


    # videos = rollout_results['rollout_videos']
    # del rollout_results['rollout_videos']
    with open(os.path.join(save_dir, 'data.json'), 'w') as f:
        json.dump(rollout_results, f)

    


if __name__ == "__main__":
    main()