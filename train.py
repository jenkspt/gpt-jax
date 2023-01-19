from typing import Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
import wandb
import numpy as np
import tyro

import jax
import jax.numpy as jnp
import flax
from flax.core import FrozenDict, frozen_dict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.jax_utils import replicate, unreplicate
import optax

from model import GPT, GPTConfig


@dataclass(frozen=True)
class WandbConfig:
    entity: str = 'jenkspt'
    project: str = 'owt'
    name: str = 'test'


@dataclass(frozen=True)
class CosineDecayScheduleConfig:
    init_value: float = 0.0
    peak_value: float = 2.5e-4
    warmup_steps: int = 2000
    decay_steps: int = 320000
    end_value: float = 1e-5


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 555
    out_dir: str = 'out'
    train_path: Path = Path('train.bin')
    val_path: Path = Path('val.bin')
    eval_interval: int = 500
    #log_interval: int = 1
    eval_iters: int = 50
    #eval_only: bool = False # if True, script exits right after the first eval
    # data
    #dataset: str = 'openwebtext'
    batch_size: int = 2
    # adamw optimizer
    max_steps: int = 500000 # total number of training iterations
    shuffle_buffer_size: int = 64
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    learning_rate: Union[float, CosineDecayScheduleConfig] = field(default_factory=CosineDecayScheduleConfig)
    # wandb logging
    wandb: Optional[WandbConfig] = field(default_factory=WandbConfig)
    # model
    model: GPTConfig = field(default_factory=GPTConfig)


@partial(jax.pmap, axis_name='batch')
def train_step(state: TrainState, tokens: jnp.ndarray, rngs: dict) -> Tuple[jnp.ndarray, TrainState]:

    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        X, Y = tokens[:, :-1], tokens[:, 1:]
        logits = state.apply_fn(params, X, False, rngs=rngs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()
        return loss
    
    # per-device loss and grads
    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    #loss, grads = jax.value_and_grad(loss_fn, has_aux=False, reduce_axes=('batch',))(state.params)
    # average gradients across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


@partial(jax.pmap, axis_name='batch')
def eval_step(state: TrainState, tokens: jnp.ndarray, rngs: dict) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(state.params, X, True, rngs=rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss


def evaluate(state: TrainState, data: np.memmap, batch_size: int, block_size: int, max_steps: int, key) -> jnp.ndarray:

    # we want a different dropout rng key for each device
    keys = jax.random.split(key, jax.device_count())
    keys_dropout = jnp.array_split(keys, jax.local_device_count())[jax.process_index()]
    losses = []

    n = jax.local_device_count() * config.batch_size * (block_size + 1)
    data = data[:(len(data) // n) * n].reshape(-1, jax.local_device_count(), batch_size, block_size + 1)

    for _, tokens in zip(range(max_steps), data):
        loss = eval_step(state, tokens, {'dropout': keys_dropout})
        losses.append(loss)
    return jnp.mean(jnp.stack(losses))


def count_params(params: FrozenDict) -> int:
    p = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)


def param_decay_mask(params: FrozenDict) -> FrozenDict:
    """ pytree mask for non-bias parameters """
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_param_mask = {k: k[-1] not in ('bias', 'embedding', 'scale') for k in flat_params.keys()}
    param_mask = flax.traverse_util.unflatten_dict(flat_param_mask)
    return frozen_dict.freeze(param_mask)


def get_train_batch(data, batch_size, block_size, key):
    ix = jax.random.randint(key, [jax.local_device_count() * batch_size], 0, len(data) - block_size)
    tokens = jnp.stack([data[i:i+block_size+1] for i in ix])
    return tokens.reshape(-1, batch_size, block_size+1)


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    if config.wandb is not None:
        wandb.init(**asdict(config.wandb))
        wandb.config = asdict(config)

    block_size = config.model.block_size

    # ===== datasets =====
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(config.train_path, dtype=np.uint16, mode='r')

    # ===== learning rate schedule =====
    if isinstance(config.learning_rate, CosineDecayScheduleConfig):
        learning_rate = optax.warmup_cosine_decay_schedule(**asdict(config.learning_rate))
    else:
        learning_rate = config.learning_rate

    # =====  init parameters ============
    key = jax.random.PRNGKey(config.seed)
    key, key_params = jax.random.split(key, 2)

    model = GPT(config.model)
    params = model.init(key_params)

    num_params = count_params(params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

    optimizer = optax.chain(
        # Apply weight decay only to non-bias parameters
        optax.add_decayed_weights(config.weight_decay, mask=param_decay_mask(params)),
        optax.adamw(learning_rate, *config.betas, weight_decay=0.0),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    del params

    step = 0
    best_val_loss = float('inf')

    # ==== restore dataset and train state ==== #
    # restore unreplicated optimizer + model state from last checkpoint.
    # this is a no-op if no checkpoints exist
    train_state, key = checkpoints.restore_checkpoint(
        f'{config.out_dir}/checkpoints/train_state', (train_state, key))

    # grab step from last checkpoint
    step = int(train_state.step)

    # replicate parameters to each device
    train_state = replicate(train_state)

    for step in range(step, config.max_steps):

        if step % config.eval_interval == 0:
            key, key_eval = jax.random.split(key)
            val_loss = evaluate(train_state, val_data, config.batch_size,
                                block_size, config.eval_iters, key_eval)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save train state in process 0
                checkpoints.save_checkpoint_multiprocess(
                    f'{config.out_dir}/checkpoints/train_state',
                    (unreplicate(train_state), key), step, keep=3)
                
                if (config.wandb is not None) and (jax.process_index() == 0):
                    wandb.log({"val/loss": val_loss}, step=step)
                print('done eval')

        key, key_batch = jax.random.split(key)
        tokens = get_train_batch(train_data, config.batch_size, block_size, key_batch)

        keys = jax.random.split(key, jax.device_count()+1)
        key = keys[0]   # new key state
        # keys for train step
        keys_dropout = jnp.array_split(keys[1:], jax.process_count())[jax.process_index()]
        loss, train_state = train_step(train_state, tokens, {'dropout': keys_dropout})

        if (config.wandb is not None) and (jax.process_index() == 0):
            wandb.log({
                "train/loss": loss,
                "lr": learning_rate(step) if callable(learning_rate) else learning_rate
            }, step=step)