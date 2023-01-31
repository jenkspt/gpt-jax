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
from train import (
    train_step,
    init_train_state,
    get_train_batch,
    count_params,
    param_decay_mask,
)

# -----------------------------------------------------------------------------

def benchmark(
    batch_size: int = 8,
    seed: int = 1337,
    model: GPTConfig = GPTConfig()):

    block_size = model.block_size

    key = jax.random.PRNGKey(seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    keys_dropout = jax.random.split(key_dropout, jax.device_count())
    # make sure dropout keys are different for each device (local and global)
    keys_dropout = jnp.array_split(keys_dropout, jax.process_count())[jax.process_index()]

    optimizer = optax.adam(3e-4, 0.9, 0.95)

    train_state = init_train_state(key_params, model, optimizer)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

    train_state = replicate(train_state)

    data = np.memmap(f'train_{jax.process_index()}.bin', dtype=np.uint16, mode='r')
    tokens = get_train_batch(data, batch_size, block_size, key)
    # simple benchmarking
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), train_state.params)
        t0 = time.time()
        for k in range(num_steps):
            key, key_batch = jax.random.split(key)
            #tokens = get_train_batch(data, batch_size, block_size, key_batch)
            loss, train_state = train_step(train_state, tokens, keys_dropout)

            #print(f"{k}/{num_steps} loss: {loss:.4f}")
        loss.block_until_ready()
        t1 = time.time()
        if stage == 1:
            print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")
            print(f"time per block: {(t1-t0)/num_steps*1000/batch_size/jax.device_count():.4f}ms")

if __name__ == "__main__":
    tyro.cli(benchmark)