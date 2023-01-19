import time
import numpy as np
import tyro

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate
import optax

from model import GPT, GPTConfig
from train import train_step, get_train_batch, count_params, param_decay_mask

# -----------------------------------------------------------------------------

def benchmark(
    batch_size: int = 8,
    block_size: int = 1024,
    seed: int = 1337,
    model: GPTConfig = GPTConfig()):

    key = jax.random.PRNGKey(seed)
    key, key_params = jax.random.split(key, 2)

    model = GPT(model)
    params = model.init(key_params)

    num_params = count_params(params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

    optimizer = optax.chain(
        # Apply weight decay only to non-bias parameters
        optax.add_decayed_weights(1e-2, mask=param_decay_mask(params)),
        optax.adamw(3e-4, weight_decay=0.0),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)

    train_state = replicate(train_state)
    del params

    data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    # simple benchmarking
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), train_state.params)
        t0 = time.time()
        for k in range(num_steps):
            key, key_batch = jax.random.split(key)
            tokens = get_train_batch(data, batch_size, block_size, key_batch)
            keys = jax.random.split(key, jax.device_count()+1)
            key = keys[0]   # new key state
            # keys for train step
            keys_dropout = jnp.array_split(keys[1:], jax.process_count())[jax.process_index()]
            loss, train_state = train_step(train_state, tokens, {'dropout': keys_dropout})

            #print(f"{k}/{num_steps} loss: {loss:.4f}")
        loss.block_until_ready()
        t1 = time.time()
        if stage == 1:
            print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")

if __name__ == "__main__":
    tyro.cli(benchmark)