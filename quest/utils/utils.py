import copy
import json
import os
import random
from pathlib import Path
import quest.utils.tensor_utils as TensorUtils
import numpy as np
import torch
import torch.nn as nn

def get_experiment_dir(cfg):
    # if eval_flag:
    #     prefix = "evaluations"
    # else:
    #     prefix = "experiments"
    #     if cfg.pretrain_model_path != "":
    #         prefix += "_finetune"


    experiment_dir = (
        f"{cfg.output_prefix}/{cfg.task.benchmark_name}/{cfg.task.sub_benchmark_name}/"
        + f"{cfg.algo.name}/{cfg.exp_name}"
    )

    if cfg.make_unique_experiment_dir:
        # look for the most recent run
        experiment_id = 0
        if os.path.exists(experiment_dir):
            for path in Path(experiment_dir).glob("run_*"):
                if not path.is_dir():
                    continue
                try:
                    folder_id = int(str(path).split("run_")[-1])
                    if folder_id > experiment_id:
                        experiment_id = folder_id
                except BaseException:
                    pass
            experiment_id += 1

        experiment_dir += f"/run_{experiment_id:03d}"
    else:
        experiment_dir += f'/stage_{cfg.stage}'
        
        assert not os.path.exists(experiment_dir), \
            f'cfg.make_unique_experiment_dir=false but {cfg.make_unique_experiment_dir} is already occupied'

    experiment_name = "_".join(experiment_dir.split("/")[len(cfg.output_prefix.split('/')):])
    return experiment_dir, experiment_name


def map_tensor_to_device(data, device):
    """Move data to the device specified by device."""
    return TensorUtils.map_tensor(
        data, lambda x: safe_device(x, device=device)
    )

def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()


def extract_state_dicts(inp):

    if not (isinstance(inp, dict) or isinstance(inp, list)):
        if hasattr(inp, 'state_dict'):
            return inp.state_dict()
        else:
            return inp
    elif isinstance(inp, list):
        out_list = []
        for value in inp:
            out_list.append(extract_state_dicts(value))
        return out_list
    else:
        out_dict = {}
        for key, value in inp.items():
            out_dict[key] = extract_state_dicts(value)
        return out_dict
        

def save_state(state_dict, path):
    save_dict = extract_state_dicts(state_dict)
    torch.save(save_dict, path)

def load_state(path):
    return torch.load(path)

def torch_save_model(model, optimizer, scheduler, model_path, cfg=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg,
        },
        model_path,
    )

def torch_load_model(model_path):
    checkpoint = torch.load(model_path)
    return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["scheduler_state_dict"], checkpoint["cfg"]

# def get_task_names(benchmark_name, sub_benchmark_name):
#     if benchmark_name == "metaworld":
#         if sub_benchmark_name == "ML45":
#             return ML45
#         elif sub_benchmark_name == "ML5":
#             return ML5
#         else:
#             raise ValueError(f"Unknown sub_benchmark_name {sub_benchmark_name}")
#     else:
#         raise ValueError(f"Unknown benchmark name {benchmark_name}")
    
# ML45 = [     
#         'assembly-v2',
#         'basketball-v2',
#         'button-press-topdown-v2',
#         'button-press-topdown-wall-v2',
#         'button-press-v2',
#         'button-press-wall-v2',
#         'coffee-button-v2',
#         'coffee-pull-v2',
#         'coffee-push-v2',
#         'dial-turn-v2',
#         'disassemble-v2',
#         'door-close-v2',
#         'door-open-v2',
#         'drawer-close-v2',
#         'drawer-open-v2',
#         'faucet-close-v2',
#         'faucet-open-v2',
#         'hammer-v2',
#         'handle-press-side-v2',
#         'handle-press-v2',
#         'handle-pull-side-v2',
#         'handle-pull-v2',
#         'lever-pull-v2',
#         'peg-insert-side-v2',
#         'peg-unplug-side-v2',
#         'pick-out-of-hole-v2',
#         'pick-place-v2',
#         'pick-place-wall-v2',
#         'plate-slide-back-side-v2',
#         'plate-slide-back-v2',
#         'plate-slide-side-v2',
#         'plate-slide-v2',
#         'push-back-v2',
#         'push-v2',
#         'push-wall-v2',
#         'reach-v2',
#         'reach-wall-v2',
#         'shelf-place-v2',
#         'soccer-v2',
#         'stick-pull-v2',
#         'stick-push-v2',
#         'sweep-into-v2',
#         'sweep-v2',
#         'window-close-v2',
#         'window-open-v2',
# ]

# ML5 = [
#     "bin-picking-v2",
#     "handle-press-side-v2",
#     "peg-unplug-side-v2",
#     "box-close-v2",
#     "hand-insert-v2",
# ]