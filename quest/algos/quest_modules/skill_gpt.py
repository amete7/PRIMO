from torch import nn
import torch
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from quest.algos.utils.mlp_proj import MLPProj



class SkillGPT(nn.Module):
    def __init__(self,
                 action_dim,
                 start_token,
                 offset_layers,
                 offset_hidden_dim,
                 offset_dim,
                 vocab_size,
                 block_size,
                 n_layer,
                 n_head,
                 n_embd,
                 attn_pdrop,
                 embd_pdrop,
                 beam_size, # value of k for top k sampling
                 temperature, # temperature for sampling
                 device,
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.start_token = start_token
        self.block_size = block_size
        self.n_embd = n_embd
        self.beam_size = beam_size
        self.temperature = temperature
        self.device = device

        self.tok_emb = nn.Embedding(vocab_size+1, n_embd)
        self.add_positional_emb = Summer(PositionalEncoding1D(n_embd))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=attn_pdrop,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layer
        )
        self.head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(embd_pdrop)
        self.lnf = nn.LayerNorm(n_embd)
        self.return_offset = offset_layers > 0
        if self.return_offset:
            self.offset_head = MLPProj(n_embd, offset_dim, hidden_size=offset_hidden_dim, num_layers=offset_layers)

    def forward(self, idx, context, targets=None):
        x = self.tok_emb(idx)
        x = self.add_positional_emb(x)
        x = torch.cat([context, x], dim=1)
        x = self.drop(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),x.device)
        x = self.decoder(x, mask=mask, is_causal=True)
        x = x[:, context.size(1):, :]
        x = self.lnf(x)
        logits = self.head(x)
        
        offset = self.offset_head(x[:,-1,:]) if self.return_offset else None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss
            return logits, loss, offset
        else:
            return logits, offset
        
    def get_indices_top_k(self, context, codebook_size, vae_block_size):
        x = torch.ones((context.shape[0], 1)).long().to(self.device)*self.start_token
        for i in range(self.block_size):
            if i == self.block_size-1:
                logits, offset = self.forward(x, context)
                logits = logits[:,:,:codebook_size]
                offset = offset.view(-1, vae_block_size, self.action_dim) if self.return_offset else None
            else:
                logits,_ = self.forward(x, context)
                logits = logits[:,:,:codebook_size]
            next_indices = top_k_sampling(logits[:,-1,:], self.beam_size, self.temperature)
            x = torch.cat([x, next_indices], dim=1)
        return x[:,1:], offset
    
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


