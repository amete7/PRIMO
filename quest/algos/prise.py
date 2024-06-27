from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as T
# from torch.nn.parallel import DistributedDataParallel as DDP
# from quest.algos.baseline_modules.prise_utils.data_augmentation import BatchWiseImgColorJitterAug, TranslationAug, DataAugGroup
import quest.algos.baseline_modules.prise_utils.misc as utils
from quest.algos.baseline_modules.prise_modules import SinusoidalPositionEncoding, TokenPolicy, Autoencoder
from quest.algos.base import Policy
import quest.utils.tensor_utils as TensorUtils
import itertools

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
            n_code, 
            alpha, 
            decoder_type, 
            decoder_loss_coef,
            device, 
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
        self.n_code = n_code
        self.alpha  = alpha
        self.decoder_type = decoder_type
        self.decoder_loss_coef = decoder_loss_coef
        self.positional_embedding = SinusoidalPositionEncoding(feature_dim)
        self.stage = stage
        self.optimizer_factory = optimizer_factory
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
        
        # self.train()

    def get_optimizers(self):
        if self.stage == 0:
            return [self.optimizer_factory(params=self.autoencoder.parameters())]
        elif self.stage == 1:
            pass
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
            self.mask = utils.generate_causal_mask(time_step, num_modalities).to(self.device)
        z_history = self.autoencoder.transformer_embedding(z_history,mask=self.mask)
        z_history = z_history.reshape(batch_size, time_step, num_modalities, feature_dim)
        return z_history[:, -1, 0]


    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)

    def compute_autoencoder_loss(self, data):

        metrics = dict()
        
        assert data['obs']['corner_rgb'].shape[1] == self.frame_stack + self.future_obs

        ### Convert inputs into tensors
        data = self.preprocess_input(data)
        obs_emb = self.obs_encode(data, reduction='stack')
        action_seq = data['actions'][:, self.frame_stack:]

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
            decoder_loss += d_loss * self.decoder_loss_coef
        
            ### Calculate embedding of next timestep
            z = self.autoencoder.transition(z+u_quantized)
            # next_obs = next_img[:, k], next_state[:, k]
            next_z = z_future[:, k:k+1] ### (batch_size, 1, 4, feature_dim)
            
            ### Calculate Dynamics loss and update latent history with the latest timestep
            z_history = torch.concatenate([z_history[:, 1:], next_z], dim=1)
            o_embed   = self.compute_transformer_embedding(z_history)
            y_next    = self.autoencoder.proj_s(o_embed).detach()
            y_pred = self.autoencoder.predictor(self.autoencoder.proj_s(z)) 
            dynamics_loss += utils.dynamics_loss(y_pred, y_next)
        
        loss = dynamics_loss + decoder_loss + quantize_loss
        info = {
            'autoencoder/total_loss': loss.item(),
            'autoencoder/dynamics_loss': dynamics_loss.item(),
            'autoencoder/quantize_loss': quantize_loss.item(),
            'autoencoder/decoder_loss': quantize_loss.item(),
        }
        return loss, info

        # self.prise_opt.zero_grad()
        # (dynamics_loss + decoder_loss + quantize_loss).backward()
        # self.prise_opt.step()
        # metrics['dynamics_loss'] = dynamics_loss.item()
        # metrics['quantize_loss'] = quantize_loss.item()
        # metrics['decoder_loss']  = decoder_loss.item()
        # return metrics
        
    def get_action(self, obs):
        pass
    
    def update(self, replay_iter, step):
        metrics = dict()
        batch = next(replay_iter)
        obs_history, action, action_seq, next_obs = batch
        metrics.update(self.update_prise(obs_history, action, action_seq, next_obs))
        return metrics
    
    def downstream_adapt(self, replay_iter, tok_to_code, 
                          tok_to_idx, idx_to_tok, finetune_decoder=True):
        self.train(False)
        
        metrics = dict()
        batch = next(replay_iter)
        
        obs_history, action, tok, action_seq, next_obs = batch
        obs_history = utils.to_torch(obs_history, device=self.device)
        next_obs    = utils.to_torch(next_obs, device=self.device)
        action = torch.torch.as_tensor(action, device=self.device).float()
        action_seq = utils.to_torch(action_seq, device=self.device)
        index = torch.tensor([tok_to_idx(x) for x in tok]).long().to(self.device)
        # tok = torch.torch.as_tensor(tok, device=self.device).reshape(-1)
        
        
        
        with torch.no_grad():
            ### (batch_size, num_step_history, 3, 128, 128)
            img_history, state_history = obs_history
            ### (batch_size, num_step, 3, 128, 128)
            next_img, next_state = next_obs
            
            batch_size = next_img.shape[0]
            action_dim = action.shape[-1]
            nstep = next_img.shape[1]
            nstep_history = img_history.shape[1]
            
            ### Data Augmentation (make sure that the augmentation is consistent across timesteps)
            img_seq = torch.concatenate([img_history, next_img], dim=1)
            img_seq, = self.aug((img_seq,))
            state_seq = torch.concatenate([state_history, next_state], dim=1)
            
            z_seq = []
            for i in range(nstep):
                z = self.encode_history([img_seq[:, i:i+nstep_history],
                                         state_seq[:, i:i+nstep_history],
                                         ], aug=False)
                z_seq.append(self.compute_transformer_embedding(z)) ###(batch_size, feature_dim)
        
        meta_action = self.autoencoder.token_policy(z_seq[0])
        token_policy_loss = F.cross_entropy(meta_action, index)
        
        ###################### Finetune Action Decoder ###########################
        if finetune_decoder:
            decoder_loss_lst = []

            vocab_size = len(idx_to_tok)
            rollout_length = [min(nstep, len(tok_to_code(torch.tensor(idx_to_tok[idx])))) for idx in range(vocab_size)]
            z = torch.concatenate([z.unsqueeze(1) for z in z_seq], dim=1)
            z = z.unsqueeze(1).repeat(1, vocab_size, 1, 1) 
            action = torch.concatenate([action.unsqueeze(1) for action in action_seq], dim=1)
            action = action.unsqueeze(1).repeat(1, vocab_size, 1, 1) 

            with torch.no_grad():
                u_quantized_lst = []
                for idx in range(vocab_size):
                    u_quantized_seq = []
                    for t in range(nstep):
                        learned_code   = self.autoencoder.a_quantizer.embedding.weight
                        if t < rollout_length[idx]:
                            u_quantized    = learned_code[tok_to_code(torch.tensor(idx_to_tok[idx]))[t], :]
                        else:
                            u_quantized    = learned_code[tok_to_code(torch.tensor(idx_to_tok[idx]))[0], :]
                        u_quantized    = u_quantized.repeat(batch_size, 1)
                        u_quantized_seq.append(u_quantized.unsqueeze(1))
                    u_quantized = torch.concatenate(u_quantized_seq,dim=1) ### (batch_size, nstep, feature_dim)
                    u_quantized_lst.append(u_quantized.unsqueeze(1))
                u_quantized_lst = torch.concatenate(u_quantized_lst,dim=1) ### (batch_size, vocab_size, nstep, feature_dim)

            ### Decode the codes into action sequences and calculate L1 loss ### (batch_size*nstep*feature_dim, -1)
            decode_action = self.autoencoder.decoder((z + u_quantized_lst).reshape(-1, z.shape[-1]))
            action = action.reshape(-1, action_dim)
            if self.decoder_type == 'deterministic':
                decoder_loss = torch.sum(torch.abs(decode_action-action), dim=-1, keepdim=True)
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
        
        self.prise_opt.zero_grad()
        (token_policy_loss+self.alpha*decoder_loss).backward()
        self.prise_opt.step()
        
        metrics = dict()
        metrics['token_policy_loss'] = token_policy_loss.item()
        metrics['decoder_loss'] = decoder_loss.item()
        return metrics

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['tokenizer'] = self.tokenizer
        return state_dict
    
    # This is janky as hell I know
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.tokenizer = state_dict.pop('tokenizer')
        # if self.tokenizer is not None:
        #     self.policy = TokenPolicy(self.feature_dim, self.hidden_dim, self.tokenizer.vocab_size)
        return super().load_state_dict(state_dict, strict, assign)