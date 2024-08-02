import torch
import torch.nn.functional as F
import numpy as np
import quest.utils.tensor_utils as TensorUtils
import itertools
import time
from quest.algos.base import ChunkPolicy


class QueST(ChunkPolicy):
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 stage,
                 optimizer_factory,
                 scheduler_factory,
                 image_encoder_factory,
                 proprio_encoder,
                 obs_proj,
                 task_encoder,
                 image_aug,
                 loss_fn,
                 l1_loss_scale,
                 action_horizon,
                 shape_meta, 
                 device,
                 do_fewshot_embedding_hack=False
                 ):
        super().__init__(
            image_encoder_factory, 
            proprio_encoder, 
            obs_proj, 
            image_aug, 
            shape_meta, 
            action_horizon,
            device)
        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.task_encoder = task_encoder

        self.start_token = self.policy_prior.start_token
        self.l1_loss_scale = l1_loss_scale if stage == 2 else 0
        self.vae_block_size = autoencoder.skill_block_size
        self.codebook_size = np.array(autoencoder.fsq_level).prod()
        
        self.loss = loss_fn
        
    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
                # self.optimizer_factory(params=decay),
                # self.optimizer_factory(params=no_decay, weight_decay=0.),
                # TODO: unhardcode
                # self.optimizer_factory(params=decoder_decay, lr=0.00001),
                # self.optimizer_factory(params=decoder_no_decay, weight_decay=0., lr=0.00001),

            ]
            return optimizers
            
    def get_schedulers(self, optimizers):
        return [self.scheduler_factory(optimizer=optimizer) for optimizer in optimizers]

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        task_emb = self.task_encoder(data["task_id"]).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss = self.autoencoder(data["actions"])
        recon_loss = self.loss(pred, data["actions"])
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss
            
        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
        }
        return loss, info

    def compute_prior_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        
        with torch.no_grad():
            indices = self.autoencoder.get_indices(data["actions"]).long()
        context = self.get_context(data)
        start_tokens = (torch.ones((context.shape[0], 1), device=self.device, dtype=torch.long) * self.start_token)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        
        logits, offset = self.policy_prior(x, context)
        prior_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        with torch.no_grad():
            logits = logits[:,:,:self.codebook_size]
            # print(logits)
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
            sampled_indices = sampled_indices.view(-1,logits.shape[1])
        
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        
        if offset is not None:
            offset = offset.view(*pred_actions.shape)
            pred_actions = pred_actions + offset
        
        l1_loss = self.loss(pred_actions, data["actions"])

        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        info = {
            'loss': total_loss.item(),
            'nll_loss': prior_loss.item(),
            'l1_loss': l1_loss.item()
        }
        return total_loss, info

    def sample_actions(self, data):
        #start_time = time.time()
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices, offset = self.policy_prior.get_indices_top_k(context, self.codebook_size, self.vae_block_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        #print(time.time() - start_time)
        pred_actions_with_offset = pred_actions + offset if offset is not None else pred_actions
        pred_actions_with_offset = pred_actions_with_offset.permute(1,0,2)
        return pred_actions_with_offset.detach().cpu().numpy()
