import metaworld
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

    model = SkillGPT_Model(cfg, shape_meta).to(device)

    if cfg.benchmark_name == "metaworld":
        ml45 = metaworld.ML45()
        if cfg.sub_benchmark_name == "ML45":
            class_dict = ml45.train_classes
            task_list = ml45.train_tasks
        elif cfg.sub_benchmark_name == "ML5":
            class_dict = ml45.test_classes
            task_list = ml45.test_tasks
        else:
            raise ValueError(f"Unknown sub_benchmark_name {cfg.sub_benchmark_name}")
    else:
        raise ValueError(f"Unknown benchmark_name {cfg.benchmark_name}")
    
    evals_per_task = cfg.eval.evals_per_task
    max_steps = cfg.eval.max_steps_per_episode
    success_rates = {}
    for name, env_cls in class_dict.items():
        print(f"Running evaluation for {name}")
        task_id = task_names.index(name)
        env = env_cls(render_mode='rgb_array', camera_name='corner')
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        tasks = [task for task in task_list if task.env_name == name]
        # sample n tasks from above list
        tasks = random.choices(tasks, k=evals_per_task)
        completed = 0
        for task in tqdm(tasks):
            success = run_episode(env, task, model, task_id, max_steps, seed)
            if success:
                completed += 1
        success_rates.update({name: completed/len(tasks)})
        print(f"Success rate for {name}: {completed/len(tasks)}")
    print(success_rates)
    print("Average success rate: ", sum(success_rates.values())/len(success_rates))

def run_episode(env, task, policy, task_id, max_steps=500, seed=42):
    env.set_task(task)
    env.seed(seed)
    obs, _ = env.reset()
    obs_input = get_data(obs, env.render(), task_id)
    success = False
    count = 0
    while count < max_steps and not success:
        count += 1
        action = policy.get_action(obs_input)
        action = np.squeeze(action, axis=0)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, r, _, _, info = env.step(action.copy())
        print(r)
        obs_input = get_data(next_obs, env.render(), task_id)
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