from dataclasses import dataclass

import torch
import torch.nn as nn

import torch.nn.functional as F


@dataclass
class TransformerConfig:
    dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    ff_dim: int = 3072
    layers_num: int = 6
    block_size: int = 8192
    vocab_size: int = 50257 + 1


class MLP(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, ff_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ff_dim, dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout_p=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout_p

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.qkv_linear(x).chunk(3, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.out_proj(attn_output)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, head_dim, ff_dim):
        super().__init__()
        
        self.attention = Attention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        attn_output = self.attention(x)
        # x = self.norm1(x + attn_output)
        x += attn_output

        ffn_output = self.mlp(x)
        # x = self.norm2(x + ffn_output)
        x += ffn_output

        return x


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.dim),
            wpe = nn.Embedding(config.block_size, config.dim),

            layers = nn.ModuleList([TransformerLayer(config.dim, config.num_heads, config.head_dim, config.ff_dim)
                                    for _ in range(config.layers_num)]),

            lm_head = nn.Linear(config.dim, config.vocab_size) # TODO: Make this shared
        ))


    def forward(self, x):
        batch_size, block_size = x.shape

        x = self.transformer.wte(x) + \
            self.transformer.wpe(torch.arange(block_size, device=x.device).repeat((batch_size, 1))) # TODO: Replace with rotary

        for layer in self.transformer.layers:
            x = layer(x)

        logits = self.transformer.lm_head(x)

        return logits
