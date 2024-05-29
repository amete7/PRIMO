import math
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize, FSQ
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer



###############################################################################
#
# Skill-GPT module
#
###############################################################################

class SkillGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size+1, cfg.n_embd)
        self.add_positional_emb = Summer(PositionalEncoding1D(cfg.n_embd))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.n_embd,
                nhead=cfg.n_head,
                dim_feedforward=4*cfg.n_embd,
                dropout=cfg.attn_pdrop,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=cfg.n_layer
        )
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.lnf = nn.LayerNorm(cfg.n_embd)
        if cfg.offset_layers > 0:
            self.offset_head = MLP_Proj(cfg.n_embd, cfg.offset_hidden_dim, cfg.offset_dim, num_layers=cfg.offset_layers)

    def forward(self, idx, context, targets=None, return_offset=False):
        x = self.tok_emb(idx)
        x = self.add_positional_emb(x)
        x = torch.cat([context, x], dim=1)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),x.device)
        x = self.decoder(x, mask=mask, is_causal=True)
        x = x[:, context.size(1):, :]
        x = self.lnf(x)
        logits = self.head(x)
        
        offset = self.offset_head(x[:,-1,:]) if return_offset else None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss
            return logits, loss, offset
        else:
            return logits, offset
    
def top_k_sampling(logits, k, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Find the top k values and indices
    top_values, top_indices = torch.topk(scaled_logits, k, dim=-1)
    # Compute probabilities from top values
    top_probs = torch.softmax(top_values, dim=-1)
    # Sample token index from the filtered probabilities
    sampled_indices = torch.multinomial(top_probs, num_samples=1, replacement=True)
    # Map the sampled index back to the original logits tensor
    original_indices = top_indices.gather(-1, sampled_indices)
    return original_indices