import copy
from collections import OrderedDict
import os
import numpy as np
import quest.utils.file_utils as FileUtils
import quest.utils.obs_utils as ObsUtils
import quest.utils.utils as utils
from PIL import Image
from quest.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
from quest.utils.frame_stack import FrameStackObservationFixed
import torch
import torch.nn as nn
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from torch.utils.data import ConcatDataset
import gym
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from libero.libero.benchmark import get_benchmark
from transformers import AutoModel, AutoTokenizer, logging
from hydra.utils import to_absolute_path
import time
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter

class LiberoWrapper(gym.Wrapper):
    def __init__(self,
                 task_id,
                 benchmark,
                 env_num=1,
                 img_height=128,
                 img_width=128,
                 obs_modality=None,
                 obs_key_mapping=None,
                 device="cuda",):
        self.task_emb = benchmark.get_task_emb(task_id)
        self.env_num = env_num
        self.obs_modality = obs_modality
        self.obs_key_mapping = obs_key_mapping
        self.device = device
        env_args = {
            "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
            "camera_heights": img_height,
            "camera_widths": img_width,
        }
        # Try to handle the frame buffer issue
        env_creation, count = False, 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")
        super().__init__(env)

    def reset(self, init_states=None):
        assert init_states is not None and len(init_states) == self.env_num
        self.env.reset()
        obs = self.env.set_init_state(init_states)
        # TODO: is below needed?
        # dummy actions [env_num, 7] all zeros for initial physics simulation
        dummy = np.zeros((self.env_num, 7))
        for _ in range(5):
            obs, _, _, _ = self.env.step(dummy)
        obs_data = self.raw_obs_to_tensor_obs(obs)
        return obs_data

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_data = self.raw_obs_to_tensor_obs(obs)
        return obs_data, reward, done, info
    
    def raw_obs_to_tensor_obs(self, obs):
        env_num = len(obs)
        data = {
            "obs": {},
            "task_emb": self.task_emb.repeat(env_num, 1),
        }
        all_obs_keys = []
        for modality_name, modality_list in self.obs_modality.items():
            for obs_name in modality_list:
                data["obs"][obs_name] = []
            all_obs_keys += modality_list
        for k in range(env_num):
            for obs_name in all_obs_keys:
                data["obs"][obs_name].append(
                    ObsUtils.process_obs(
                        torch.from_numpy(obs[k][self.obs_key_mapping[obs_name]]),
                        obs_key=obs_name,
                    ).float()
                )
        for key in data["obs"]:
            data["obs"][key] = torch.stack(data["obs"][key])
        data = utils.map_tensor_to_device(data, self.device)
        return data

def build_dataset(data_prefix, 
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  obs_modality,
                  obs_seq_len=1, 
                  load_obs=True,
                  task_embedding_format="clip",
                  ):
    benchmark = get_benchmark(benchmark_name)
    n_tasks = benchmark.n_tasks
    few_shot_demos = [1, 5, 10, 20, 45] if mode == 'fewshot' else None
    few_shot_demos_list = [f"demo_{i}" for i in few_shot_demos] if few_shot_demos is not None else None
    
    manip_datasets = []
    descriptions = []
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for i in range(n_tasks):
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                data_prefix, benchmark.get_task_demonstration(i)
            ),
            obs_modality=obs_modality,
            initialize_obs_utils=(i == 0),
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            few_demos = few_shot_demos_list,
        )
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(task_embedding_format, descriptions)
    benchmark.set_task_embs(task_embs)
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset

def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    few_demos=None,
    ):
    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
    )
    return dataset

class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict

def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    return task_embs