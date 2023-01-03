from typing import Any, Optional
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    dtype: Optional[Any] = None


class SelfAttention(nn.MultiHeadDotProductAttention):

    @partial(nn.remat, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)
    @nn.compact
    def __call__(self, inputs_q, mask = None, deterministic = None):
        x = super().__call__(inputs_q, inputs_q, mask, deterministic)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x, deterministic)
        return x


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype)(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, mask=None, deterministic=None):
        h = nn.LayerNorm(dtype=self.config.dtype)(x)
        x = x + SelfAttention(self.config.num_heads, dtype=self.config.dtype)(
            h, mask=mask, deterministic=deterministic)
        h = nn.LayerNorm(dtype=self.config.dtype)(x)
        x = x + MLP(self.config)(h, deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        #deterministic = nn.merge_param('deterministic', self.config.deterministic, deterministic)

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')

        token_embed = wte(idx)      # [B, T, num_embeds]
        pos_embed = wpe(pos)        # [1, T, num_embeds]
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for _ in range(self.config.num_layers):
            x = Block(self.config)(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(dtype=self.config.dtype)(x)
        logits = nn.Dense(self.config.vocab_size,
                          use_bias=False,
                          dtype=self.config.dtype,
                          name='lm_head')(x)
        return logits

    def init(self, rng):
        """
        by jitting init, traced values instead of concrete values are used
        which saves memory (since un-jitted model may not fit in memory)
        """
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params
