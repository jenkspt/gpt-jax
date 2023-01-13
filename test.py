import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
from transformers import FlaxGPT2LMHeadModel
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Attention, FlaxGPT2MLP
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from model import GPT, GPTConfig, convert_hf_params

hf_config = GPT2Config(
    vocab_size=256,
    n_positions=32,
    n_embd=64,
    n_layer=1,
    n_head=2,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-6,
    use_cache=False
)

config = GPTConfig(
    vocab_size=256,
    block_size=32,
    num_embeds=64,
    num_layers=1,
    num_heads=2,
    dropout_rate=0.1,
)


def test_gpt2():
    key = jax.random.PRNGKey(0)
    key, key_idxs, key_params = jax.random.split(key, 3)

    hf_model = FlaxGPT2LMHeadModel(hf_config)
    hf_params = hf_model.init_weights(key_params, (2, 32))
    model = GPT(config)

    params = model.init(key_params)
    target_shapes = jax.tree_util.tree_map(lambda a: a.shape, params)
    params = convert_hf_params(hf_params, 2, 64)
    shapes = jax.tree_util.tree_map(lambda a: a.shape, params)

    assert shapes == target_shapes

    for k in ('ln_f', 'wpe', 'wte'):
        assert params['params'][k] == hf_params['transformer'][k]

    idxs = jax.random.randint(key_idxs, (2, 32), 0, 256)
    y1 = hf_model(idxs, params=hf_params).logits
    y2 = model.apply(params, idxs, True)
    assert jnp.allclose(y1, y2, atol=1e-6)