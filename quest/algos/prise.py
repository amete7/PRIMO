from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
# import torchvision.transforms as T
# from torch.nn.parallel import DistributedDataParallel as DDP
# from quest.algos.baseline_modules.prise_utils.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug, DataAugGroup
import quest.algos.baseline_modules.prise_utils.misc as pu
from quest.algos.baseline_modules.prise_modules import SinusoidalPositionEncoding, TokenPolicy, Autoencoder
from quest.algos.base import Policy
import quest.utils.tensor_utils as TensorUtils
from quest.algos.baseline_modules.prise_utils.tokenizer_api import Tokenizer
import quest.utils.utils as utils
import itertools
import numpy as np
from tqdm import tqdm

from quest.utils.utils import map_tensor_to_device
import quest.utils.obs_utils as ObsUtils

class PRISE(Policy):
    def __init__(
            self, 
            # obs_shape, 
            # action_dim, 
            autoencoder: Autoencoder,
            policy,

            image_encoder_factory, 
            proprio_encoder, 
            obs_proj, 
            image_aug, 
            shape_meta, 
            stage,
            optimizer_factory, 
            feature_dim,
            hidden_dim,
            frame_stack,
            future_obs,
            tokenizer_config,
            alpha, 
            decoder_type, 
            decoder_loss_coef,
            device, 
            debug_mode=False,
        ):
        super().__init__(
            image_encoder_factory, 
            proprio_encoder, 
            obs_proj, 
            image_aug, 
            shape_meta, 
            device)
        self.autoencoder = autoencoder
        # self.policy = nn.Sequential(
        #     nn.Linear(feature_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, todo)
        # ).to(self.device)
        self.policy = policy
        self.tokenizer = None

        self.device = device
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.frame_stack = frame_stack
        self.future_obs = future_obs
        self.tokenizer_config = tokenizer_config
        self.alpha  = alpha
        self.decoder_type = decoder_type
        self.decoder_loss_coef = decoder_loss_coef
        self.positional_embedding = SinusoidalPositionEncoding(feature_dim)
        self.stage = stage
        self.optimizer_factory = optimizer_factory
        self.debug_mode = debug_mode
        # self.encoders = torch.nn.ModuleList([
        #                                      Encoder(obs_shape, feature_dim),
        #                                      nn.Sequential(
        #                                         nn.Linear(8, feature_dim),
        #                                     ),
        #                                     ])
                                    
        
        # self.autoencoder = PRISE(feature_dim, action_dim, hidden_dim, self.encoders, n_code, 
        #                        device, decoder_type, decoder_loss_coef).to(device)
        # self.prise_opt = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        
        # translation_aug = TranslationAug(input_shape=obs_shape, translation=4)
        # self.aug = DataAugGroup((translation_aug,))
        self.mask = None ### will be used later for transformer masks
        self.code_buffer = None
        self.tok_to_idx = None
        self.idx_to_tok = None
        
        # self.train()

    def get_optimizers(self):
        if self.stage == 0:
            return [self.optimizer_factory(params=self.autoencoder.parameters())]
        elif self.stage == 1:
            return [self.optimizer_factory(params=self.parameters())]
        elif self.stage == 2:
            pass

    ### obs_history: (obs_agent_history, obs_wrist_history, state_history, task_embedding_history)
    ### obs_agent_history: (batch_size, time_step, 3, 128, 128)
    ### state_history: (batch_size, time_step, 9)
    # def encode_history(self, obs_history, aug=True):
    #     img_history, state_history = obs_history
    #     batch_size, time_step, num_channel, img_size = img_history.shape[0], img_history.shape[1], img_history.shape[2], img_history.shape[3]
        
    #     if aug:
    #         img_history, = self.aug((img_history,))
        
    #     img_history = img_history.reshape(-1, num_channel, img_size, img_size)
    #     z_img_history   = self.encoders[0](img_history.float())
        
    #     state_history     = state_history.reshape(-1, state_history.shape[-1])
    #     z_state_history   = self.encoders[1](state_history.float())
        
    #     z_img_history  = z_img_history.reshape(batch_size,  time_step, 1, -1)
    #     z_state_history  = z_state_history.reshape(batch_size,  time_step, 1, -1)  
    #     z_history = torch.concatenate([z_img_history, z_state_history], dim=2)
    #     return z_history

    def compute_transformer_embedding(self, z_history, reset_mask=False):
        batch_size, time_step, num_modalities, feature_dim = z_history.shape
        
        ### Add positional embedding
        positional_embedding = self.positional_embedding(z_history).to(self.device)
        z_history += positional_embedding.unsqueeze(1)
        z_history = z_history.reshape(batch_size, time_step*num_modalities, feature_dim)
        
        if self.mask is None or reset_mask:
            self.mask = pu.generate_causal_mask(time_step, num_modalities).to(self.device)
        z_history = self.autoencoder.transformer_embedding(z_history,mask=self.mask)
        z_history = z_history.reshape(batch_size, time_step, num_modalities, feature_dim)
        return z_history[:, -1, 0]

    def train_bpe(self, dataset: ConcatDataset, use_tqdm=True):
        # TODO: this funtion assumes that the dataset it receives is a torch concat dataset composed
        # of SequenceVL datasets

        self.train(False)
        lst_traj = []
        # for task_dir in self.pretraining_data_dirs:
        #     lst_traj.extend(utils.choose(list(sorted(task_dir.glob('*.npz'))), self.cfg.max_traj_per_task))
        # print('Loaded {} trajectories'.format(len(lst_traj)))
        
        with torch.no_grad():
            corpus, counter = [], 0
            for sub_dataset in tqdm(dataset.datasets, disable=not use_tqdm):
                for ep in range(sub_dataset.n_demos):
                    sequence_dataset = sub_dataset.sequence_dataset
                    actions = torch.tensor(sequence_dataset.get_dataset_for_ep(f'demo_{ep}', 'actions'), device=self.device)
                    obs = {}
                    for key in sequence_dataset.obs_keys:
                        obs[key] = torch.tensor(np.array(sequence_dataset.get_dataset_for_ep(f'demo_{ep}', f'obs/{key}')), device=self.device)
                    episode = {
                        'actions': actions,
                        'obs': obs
                    }
                    episode = self.preprocess_input(episode, train_mode=False)
                    episode['obs'] = pu.compute_traj_latent_embedding(episode, self.frame_stack, sequence_dataset.obs_keys)
                    obs_emb = self.obs_encode(episode, reduction='stack', hwc=True)
                    z = self.compute_transformer_embedding(obs_emb)
                    latents = self.autoencoder.action_encoder(z, actions)
                    _, _, _, _, codes = self.autoencoder.a_quantizer(latents) 
                    codes = list(codes.reshape(-1).detach().cpu().numpy())
                    codes = [int(idx) for idx in codes]
                    corpus.append(codes)
                    counter += 1

            print('=========Offline Data Tokenized!==========')

            ### Train tokenizer on the tokenized pretraining trajectories
            self.tokenizer = Tokenizer(algo='bpe', vocab_size=self.tokenizer_config.vocab_size)
            self.tokenizer.train(corpus, 
                                 min_frequency=self.tokenizer_config.min_frequency, 
                                 max_token_length=self.tokenizer_config.max_token_length, 
                                 verbose=True)

    def preprocess_dataset(self, dataset, use_tqdm=True):
        if self.stage == 0:
            return
        # TODO: this funtion assumes that the dataset it receives is a torch concat dataset composed
        # of SequenceVL datasets
        # Also this is pretty hacky. Not sure what a better way to do this would be 

        self.train(False)

        self.tok_to_idx = {}
        self.idx_to_tok = []        
        with torch.no_grad():
            for sub_dataset in tqdm(dataset.datasets, disable=not use_tqdm):
                for ep in range(sub_dataset.n_demos):
                    sequence_dataset = sub_dataset.sequence_dataset
                    actions = torch.tensor(sequence_dataset.get_dataset_for_ep(f'demo_{ep}', 'actions'), device=self.device)
                    if self.debug_mode:
                        codes = list(np.random.randint(0, 10, actions.shape[0]))
                    else:
                        obs = {}
                        for key in sequence_dataset.obs_keys:
                            obs[key] = torch.tensor(np.array(sequence_dataset.get_dataset_for_ep(f'demo_{ep}', f'obs/{key}')), device=self.device)
                        episode = {
                            'actions': actions,
                            'obs': obs
                        }
                        episode = self.preprocess_input(episode, train_mode=False)
                        episode['obs'] = pu.compute_traj_latent_embedding(episode, self.frame_stack, sequence_dataset.obs_keys)
                        obs_emb = self.obs_encode(episode, reduction='stack', hwc=True)
                        z = self.compute_transformer_embedding(obs_emb)
                        latents = self.autoencoder.action_encoder(z, actions)
                        _, _, _, _, codes = self.autoencoder.a_quantizer(latents) 
                        codes = list(codes.reshape(-1).detach().cpu().numpy())
                    codes = [int(idx) for idx in codes]
                    # breakpoint()
                    # print(len(codes))
                    # for t in range(len(codes)):
                    #     sliced = codes[t:]
                    #     # print(t)
                    #     print(t, len(self.tokenizer.encode(sliced, verbose=False)), sliced)
                    #     self.tokenizer.encode(sliced, verbose=False)[0]
                    traj_tok = [self.tokenizer.encode(codes[t:], verbose=False)[0] for t in range(len(codes))]
                    # episode['token'] = traj_tok
                    sequence_dataset.hdf5_cache[f'demo_{ep}']['tokens'] = np.array(traj_tok)
                    sequence_dataset.dataset_keys += ('tokens',)
                    # Set up token to index, index to token mapping
                    for tok in traj_tok:
                        if not tok in self.tok_to_idx:
                            self.tok_to_idx[tok] = len(self.tok_to_idx)
                            self.idx_to_tok.append(tok)
                    
                    # break
            
        # self.tok_to_code = lambda tok: self.tokenizer.decode([int(tok.item())], verbose=False) ### Token =>  First Code
        # self.tok_to_idx  = lambda tok: tok_to_idx[int(tok.item())] ### Token => Index
            # x = dataset[0]
            # breakpoint()
            
                    # counter += 1

        print('=========Offline Data Tokenized!==========')

            ### Train tokenizer on the tokenized pretraining trajectories
            # self.tokenizer = Tokenizer(algo='bpe', vocab_size=self.tokenizer_config.vocab_size)
            # self.tokenizer.train(corpus, 
            #                      min_frequency=self.tokenizer_config.min_frequency, 
            #                      max_token_length=self.tokenizer_config.max_token_length, 
            #                      verbose=True)

    def tok_to_code(self, tok):
        return self.tokenizer.decode([int(tok.item())], verbose=False)

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.downstream_adapt(data)

    def compute_autoencoder_loss(self, data):
        
        assert data['obs']['corner_rgb'].shape[1] == self.frame_stack + self.future_obs

        ### Convert inputs into tensors
        data = self.preprocess_input(data)
        obs_emb = self.obs_encode(data, reduction='stack')
        action_seq = data['actions'][:, self.frame_stack - 1:]

        # breakpoint()
        # obs_history = utils.to_torch(obs_history, device=self.device)
        # next_obs    = utils.to_torch(next_obs, device=self.device)
        # # action = torch.torch.as_tensor(action, device=self.device).float()
        # action_seq = utils.to_torch(action_seq, device=self.device)
        
        ### (batch_size, num_step_history, 3, 128, 128)
        # img_history, state_history = obs_history
        ### (batch_size, num_step, 3, 128, 128)
        # next_img, next_state = next_obs ### image observation, state observation
        # nstep = next_img.shape[1]
        
        ### Compute CNN-Transformer Embeddings
        z_history = obs_emb[:, :self.frame_stack]
        z_future = obs_emb[:, self.frame_stack:]
        o_embed = self.compute_transformer_embedding(z_history)
        
        z = o_embed
        dynamics_loss, quantize_loss, decoder_loss = 0, 0, 0

        for k in range(self.future_obs):
            u = self.autoencoder.action_encoder(o_embed, action_seq[:, k].float())
            q_loss, u_quantized, _, _, min_encoding_indices = self.autoencoder.a_quantizer(u)
            quantize_loss += q_loss
            
            ### Calculate encoder Loss
            decode_action = self.autoencoder.decoder(o_embed + u_quantized)
            if self.decoder_type == 'deterministic':
                d_loss = F.l1_loss(decode_action, action_seq[:, k].float())
            else:
                d_loss = self.autoencoder.decoder.loss_fn(decode_action, action_seq[:, k].float())
            decoder_loss += d_loss
        
            ### Calculate embedding of next timestep
            z = self.autoencoder.transition(z+u_quantized)
            # next_obs = next_img[:, k], next_state[:, k]
            next_z = z_future[:, k:k+1] ### (batch_size, 1, 4, feature_dim)
            
            ### Calculate Dynamics loss and update latent history with the latest timestep
            # breakpoint()
            z_history = torch.concatenate([z_history[:, 1:], next_z], dim=1)
            o_embed   = self.compute_transformer_embedding(z_history)
            y_next    = self.autoencoder.proj_s(o_embed).detach()
            y_pred = self.autoencoder.predictor(self.autoencoder.proj_s(z)) 
            dynamics_loss += pu.dynamics_loss(y_pred, y_next)
        
        loss = dynamics_loss + self.decoder_loss_coef * decoder_loss + quantize_loss
        info = {
            'total_loss': loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'quantize_loss': quantize_loss.item(),
            'decoder_loss': decoder_loss.item(),
        }
        return loss, info

        # self.prise_opt.zero_grad()
        # (dynamics_loss + decoder_loss + quantize_loss).backward()
        # self.prise_opt.step()
        # metrics['dynamics_loss'] = dynamics_loss.item()
        # metrics['quantize_loss'] = quantize_loss.item()
        # metrics['decoder_loss']  = decoder_loss.item()
        # return metrics

    # def act(self, obs, code_buffer, z_history_buffer):
    #     img = obs.observation
    #     # state     = obs.state 
    #     # img = torch.torch.as_tensor(img, device=self.device).unsqueeze(0)
    #     # state     = torch.torch.as_tensor(state, device=self.device).unsqueeze(0)
    #     # z = self.agent.encode_obs((img, state), aug=False)
        
    #     ### At timestep 0, pre-fill z_history_buffer 
    #     if len(z_history_buffer) == 0:
    #         for i in range(self.cfg.nstep_history):
    #             z_history_buffer.append(z)
    #     else:
    #         z_history_buffer.append(z) 
    #     z_history = torch.concatenate(list(z_history_buffer), dim=1) ###(1,T,4,feature_dim)
    #     z_history = self.agent.compute_transformer_embedding(z_history)
        
    #      ### Query the skill policy when the buffer is empty
    #     if len(code_buffer) == 0:
    #         meta_action = self.agent.PRISE.module.token_policy(z_history).max(-1)[1]
    #         tok = self.idx_to_tok[int(meta_action.item())]
    #         code_buffer = self.tokenizer.decode([tok], verbose=False)

    #     ### Decode the first code into raw actions
    #     code_selected = code_buffer.pop(0)
    #     learned_code  = self.agent.PRISE.module.a_quantizer.embedding.weight
    #     u = learned_code[code_selected, :]
    #     action = self.agent.PRISE.module.decode(z_history, u, decoder_type=self.cfg.decoder_type)
    #     return code_buffer, action.detach().cpu().numpy()[0]

    def reset(self):
        self.code_buffer = []

    def get_action(self, obs, task_id):
        assert self.code_buffer is not None, "you need to call policy.reset() before getting actions"

        with torch.no_grad():
            for key, value in obs.items():
                if key in self.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                obs[key] = torch.tensor(value).unsqueeze(0)
            data = {}
            data["obs"] = obs
            data["task_id"] = torch.tensor([task_id], dtype=torch.long)
            data = map_tensor_to_device(data, self.device)

            data = self.preprocess_input(data, train_mode=False)
            obs_emb = self.obs_encode(data, reduction='stack')
            # TODO: in the prise repo they do this step at train but not test time
            z = self.compute_transformer_embedding(obs_emb) 
            # z = obs_emb

            if len(self.code_buffer) == 0:

            # with torch.no_grad():
                # data = self.preprocess_input(data, train_mode=False)
                # obs_emb = self.obs_encode(data, reduction='stack')
                # # TODO: in the prise repo they do this step at train but not test time
                # z = self.compute_transformer_embedding(obs_emb) 
                meta_action = self.policy(z)[:, :len(self.idx_to_tok)].max(-1)[1]
                # meta_action = self.policy(z)[:, :len(self.idx_to_tok)]
                # breakpoint()
                # breakpoint()
                # meta_action = self.policy(z).max(-1)[1]
                tok = self.idx_to_tok[int(meta_action.item())]
                self.code_buffer = self.tokenizer.decode([tok], verbose=False)
        
            code_selected = self.code_buffer.pop(0)
            learned_code  = self.autoencoder.a_quantizer.embedding.weight
            u = learned_code[code_selected]
            action = self.autoencoder.decode(z, u, decoder_type=self.decoder_type)
        return action.detach().cpu().numpy()[0]
        
    
    # def update(self, replay_iter, step):
    #     metrics = dict()
    #     batch = next(replay_iter)
    #     obs_history, action, action_seq, next_obs = batch
    #     metrics.update(self.update_prise(obs_history, action, action_seq, next_obs))
    #     return metrics
    
    def downstream_adapt(self, data, finetune_decoder=True):
        self.train(False)
        
        metrics = dict()
        # batch = next(replay_iter)
        
        ### Convert inputs into tensors
        data = self.preprocess_input(data)
        obs_emb = self.obs_encode(data, reduction='stack')
        action_seq = data['actions'][:, self.frame_stack:]
        # breakpoint()
        tokens = data['tokens'][:, 0]
        index = torch.tensor([self.tok_to_idx[x.item()] for x in tokens], device=self.device)

        ### Compute CNN-Transformer Embeddings
        # z_history = obs_emb[:, :self.frame_stack]
        # z_future = obs_emb[:, self.frame_stack:]
        # o_embed = self.compute_transformer_embedding(z_history)

        # obs_history, action, tok, action_seq, next_obs = batch
        # obs_history = pu.to_torch(obs_history, device=self.device)
        # next_obs    = pu.to_torch(next_obs, device=self.device)
        # action = torch.torch.as_tensor(action, device=self.device).float()
        # action_seq = pu.to_torch(action_seq, device=self.device)
        # index = torch.tensor([self.tok_to_idx(x) for x in tok]).long().to(self.device)
        # tok = torch.torch.as_tensor(tok, device=self.device).reshape(-1)
        
        batch_size = obs_emb.shape[0]
        action_dim = action_seq.shape[-1]
        nstep = self.future_obs
        vocab_size = len(self.idx_to_tok)
        
        
        with torch.no_grad():
            ### (batch_size, num_step_history, 3, 128, 128)
            # img_history, state_history = obs_history
            # ### (batch_size, num_step, 3, 128, 128)
            # next_img, next_state = next_obs
            
            # nstep_history = img_history.shape[1]
            
            # ### Data Augmentation (make sure that the augmentation is consistent across timesteps)
            # img_seq = torch.concatenate([img_history, next_img], dim=1)
            # img_seq = self.aug((img_seq,))
            # state_seq = torch.concatenate([state_history, next_state], dim=1)
            
            z_seq = []
            for i in range(nstep):
                # z = self.encode_history([img_seq[:, i:i+nstep_history],
                #                          state_seq[:, i:i+nstep_history],
                #                          ], aug=False)
                z = obs_emb[:, i:i+self.frame_stack]
                z_seq.append(self.compute_transformer_embedding(z)) ###(batch_size, feature_dim)
        
        # breakpoint()
        meta_action = self.policy(z_seq[0])[:, :vocab_size]
        token_policy_loss = F.cross_entropy(meta_action, index)
        
        ###################### Finetune Action Decoder ###########################
        if finetune_decoder:
            decoder_loss_lst = []

            rollout_length = [min(nstep, len(self.tok_to_code(torch.tensor(self.idx_to_tok[idx])))) for idx in range(vocab_size)]
            z = torch.concatenate([z.unsqueeze(1) for z in z_seq], dim=1)
            z = z.unsqueeze(1).repeat(1, vocab_size, 1, 1) 
            # breakpoint()
            # action = torch.concatenate([action.unsqueeze(1) for action in action_seq], dim=1)
            action = action_seq.unsqueeze(1).repeat(1, vocab_size, 1, 1) 

            with torch.no_grad():
                u_quantized_lst = []
                for idx in range(vocab_size):
                    u_quantized_seq = []
                    for t in range(nstep):
                        learned_code   = self.autoencoder.a_quantizer.embedding.weight
                        if t < rollout_length[idx]:
                            u_quantized    = learned_code[self.tok_to_code(torch.tensor(self.idx_to_tok[idx]))[t], :]
                        else:
                            u_quantized    = learned_code[self.tok_to_code(torch.tensor(self.idx_to_tok[idx]))[0], :]
                        u_quantized    = u_quantized.repeat(batch_size, 1)
                        u_quantized_seq.append(u_quantized.unsqueeze(1))
                    u_quantized = torch.concatenate(u_quantized_seq,dim=1) ### (batch_size, nstep, feature_dim)
                    u_quantized_lst.append(u_quantized.unsqueeze(1))
                u_quantized_lst = torch.concatenate(u_quantized_lst,dim=1) ### (batch_size, vocab_size, nstep, feature_dim)


            ### Decode the codes into action sequences and calculate L1 loss ### (batch_size*nstep*feature_dim, -1)
            decode_action = self.autoencoder.decoder((z + u_quantized_lst).reshape(-1, z.shape[-1]))
            action = action.reshape(-1, action_dim)
            if self.decoder_type == 'deterministic':
                decoder_loss = torch.sum(torch.abs(decode_action - action), dim=-1, keepdim=True)
            elif self.decoder_type == 'gmm':
                decoder_loss = self.autoencoder.decoder.loss_fn(decode_action, action, reduction='none')
            else:
                print('Decoder type not supported')
                raise Exception
            decoder_loss = decoder_loss.reshape(batch_size, vocab_size, nstep)

            rollout_length = torch.torch.as_tensor(rollout_length).to(self.device)
            expanded_index = rollout_length.unsqueeze(0).unsqueeze(-1).expand(batch_size, vocab_size, nstep)
            timestep_tensor = torch.arange(nstep).view(1, 1, -1).expand_as(decoder_loss).to(self.device)
            mask = timestep_tensor < expanded_index
            decoder_loss = torch.sum(decoder_loss*mask.float(), dim=-1) ### (batch_size, vocab_size)

            meta_action_dist = F.gumbel_softmax(meta_action)
            decoder_loss = torch.mean(torch.sum(decoder_loss*meta_action_dist, dim=-1))
        else:
            decoder_loss = torch.tensor(0.)
        
        # self.prise_opt.zero_grad()
        loss = token_policy_loss + self.alpha * decoder_loss
        # self.prise_opt.step()
        
        metrics = dict()
        metrics['token_policy_loss'] = token_policy_loss.item()
        metrics['decoder_loss'] = decoder_loss.item()
        return loss, metrics

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['tokenizer'] = self.tokenizer
        state_dict['tok_to_idx'] = self.tok_to_idx
        state_dict['idx_to_tok'] = self.idx_to_tok
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.tokenizer = state_dict.pop('tokenizer')
        self.tok_to_idx = state_dict.pop('tok_to_idx')
        self.idx_to_tok = state_dict.pop('idx_to_tok')
        return super().load_state_dict(state_dict, strict, assign)