import random
import time
import os
import numpy as np
import hydra
import pprint
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.utils import create_experiment_dir, map_tensor_to_device, torch_load_model, get_task_names
from robomimic.utils.obs_utils import process_frame
from utils.metaworld_dataloader import get_dataset
from primo.stage2 import SkillGPT_Model
from envs.Metaworld.metaworld import make_env

@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    pprint.pprint(cfg)
    global device
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)

    task_names = get_task_names(cfg.benchmark_name, cfg.sub_benchmark_name)
    n_tasks = len(task_names)
    cfg.n_tasks = n_tasks
    print(task_names)
    _, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.data.data_dir, f"{cfg.sub_benchmark_name}/{task_names[0]}.hdf5"
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len,
                obs_seq_len=cfg.data.obs_seq_len,
            )
    print("Shape meta: ", shape_meta)
    model = SkillGPT_Model(cfg, shape_meta).to(device)
    state_dict, _, _, _ = torch_load_model(cfg.pretrain_model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully")

    evals_per_task = cfg.eval.evals_per_task
    max_steps = cfg.eval.max_steps_per_episode
    success_rates = {}
    for task in task_names:
        print(f"Running evaluation for {task}")
        task_idx = task_names.index(task)
        env = make_env(task, seed, max_steps)
        completed = 0
        for i in tqdm(range(evals_per_task)):
            success = run_episode(env, task_idx, model)
            if success:
                completed += 1
        success_rates.update({task: completed/evals_per_task})
        print(f"Success rate for {task}: {completed/evals_per_task}")
    print(success_rates)
    print("Average success rate: ", sum(success_rates.values())/len(success_rates))

def run_episode(env, task_idx, policy):
    obs, done, success, count = env.reset(task_idx=task_idx), False, False, 0
    obs_input = get_data(obs, env.render(), task_idx)
    while not done and not success:
        count += 1
        action = policy.get_action(obs_input)
        action = np.squeeze(action, axis=0)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        action = torch.tensor(action)
        next_obs, r, done, info = env.step(action)
        obs_input = get_data(next_obs, env.render(), task_idx)
        if int(info["success"]) == 1:
            print("Success")
            success = True
    return success

def get_data(obs, image, task_id):
    batch = {}
    batch["obs"] = {}
    batch["obs"]['robot_states'] = torch.tensor((np.concatenate((obs[:4],obs[18:22]))), dtype=torch.float32).unsqueeze(0)
    image_obs = process_frame(frame=image, channel_dim=3, scale=255.)
    batch["obs"]['corner_rgb'] = torch.tensor(image_obs).unsqueeze(0)
    batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
    batch = map_tensor_to_device(batch, device)
    return batch

if __name__ == "__main__":
    main()