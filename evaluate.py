import os
import time
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
<<<<<<< HEAD
from utils.utils import create_experiment_dir, map_tensor_to_device, torch_load_model, get_task_names
from robomimic.utils.obs_utils import process_frame
from utils.metaworld_dataloader import get_dataset
from primo.stage2 import SkillGPT_Model
from envs.Metaworld import make_env

@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(hydra_cfg):
	yaml_config = OmegaConf.to_yaml(hydra_cfg)
	cfg = EasyDict(yaml.safe_load(yaml_config))
	pprint.pprint(cfg)
	global device
	device = cfg.device
	seed = cfg.seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	task_names = get_task_names(cfg.benchmark_name, cfg.sub_benchmark_name)
	n_tasks = len(task_names)
	cfg.n_tasks = 45
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
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	print("Model loaded successfully")

	global video_dir
	video_dir = None
	if cfg.save_video:
		video_dir = os.path.join(cfg.video_dir, 'metaworld')
		os.makedirs(video_dir, exist_ok=True)

	evals_per_task = cfg.eval.evals_per_task
	max_steps = cfg.eval.max_steps_per_episode
	success_rates = {}
	ind=0
	psuedo_idx = [0,27,41,28,40]
	for task in task_names:
		print(f"Running evaluation for {task}")
		task_idx = psuedo_idx[ind]
		ind += 1
		env = make_env(task, seed, max_steps)
		completed = 0
		for i in tqdm(range(evals_per_task)):
			success = run_episode(env, task_idx, model, i)
			if success:
				completed += 1
		success_rates.update({task: completed/evals_per_task})
		print(f"Success rate for {task}: {completed/evals_per_task}")
	print(success_rates)
	print("Average success rate: ", sum(success_rates.values())/len(success_rates))

def run_episode(env, task_idx, policy, i):
	obs, done, success, count = env.reset(), False, False, 0
	frames = [env.render()]
	obs_input = get_data(obs, frames[-1], task_idx)
	while not done and not success:
		count += 1
		action = policy.get_action(obs_input)
		action = np.squeeze(action, axis=0)
		action = np.clip(action, env.action_space.low, env.action_space.high)
		action = torch.tensor(action)
		next_obs, r, done, info = env.step(action)
		frames.append(env.render())
		obs_input = get_data(next_obs, frames[-1], task_idx)
		if int(info["success"]) == 1:
			# print("Success")
			success = True
	if video_dir is not None:
		imageio.mimsave(
					os.path.join(video_dir, f'{task_idx}_{i}.mp4'), frames, fps=15)
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
=======
import quest.utils.utils as utils
from pyinstrument import Profiler
from moviepy.editor import ImageSequenceClip
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)


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

    

>>>>>>> albert/refactor

if __name__ == "__main__":
    main()
