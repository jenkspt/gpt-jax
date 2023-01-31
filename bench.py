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
    count_params,
)
from dataset import get_dataset

# -----------------------------------------------------------------------------

def benchmark(
    pattern: str = "train_??.tfrecord",
    batch_size: int = 8,
    seed: int = 1337,
    model: GPTConfig = GPTConfig()):

    block_size = model.block_size

    key = jax.random.PRNGKey(seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    # make sure dropout keys are different for each device (local and global)
    key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
    keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

    optimizer = optax.adam(3e-4, 0.9, 0.95)

    train_state = init_train_state(key_params, model, optimizer)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

    train_state = replicate(train_state)

    train_ds = get_dataset(
        pattern, batch_size,
        block_size, 64,
        seed = seed)
    train_iter = iter(train_ds)

    # simple benchmarking
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), train_state.params)
        t0 = time.time()
        for k in range(num_steps):
            tokens = next(train_iter)._numpy()
            loss, train_state = train_step(train_state, tokens, keys_dropout)

            #print(f"{k}/{num_steps} loss: {loss:.4f}")
        loss.block_until_ready()
        t1 = time.time()
        if stage == 1:
            print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")
            print(f"time per block: {(t1-t0)/num_steps*1000/batch_size/jax.device_count():.4f}ms")

if __name__ == "__main__":
    tyro.cli(benchmark)