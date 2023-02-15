import time
from dataclasses import asdict
import tyro

import jax
import optax
from flax.jax_utils import replicate

from train import (
    train_step,
    init_train_state,
    count_params,
    TrainConfig,
    get_default_config,
)
from dataset import get_dataset


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config())

    block_size = config.model.block_size

    key = jax.random.PRNGKey(config.seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    # make sure dropout keys are different for each device (local and global)
    key_dropout = jax.random.fold_in(key_dropout, jax.process_index())
    keys_dropout = jax.random.split(key_dropout, jax.local_device_count())

    # ===== learning rate schedule =====
    learning_rate = optax.warmup_cosine_decay_schedule(**asdict(config.learning_rate))

    train_state = init_train_state(key_params, config, learning_rate)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

    train_state = replicate(train_state)

    train_ds = get_dataset(
        config.train_pattern, config.batch_size,
        block_size, 64,
        seed = config.seed)
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
        if stage == 1 and jax.process_index() == 0:
            print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")
            print(f"time per block: {(t1-t0)/num_steps*1000/config.batch_size/jax.device_count():.4f}ms")