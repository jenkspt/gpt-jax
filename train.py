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
from tensorflow.io import gfile

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
    train_pattern: str = 'train'
    eval_interval: int = 500
    eval_steps: int = 50
    eval_only: bool = False # if True, script exits right after the first eval
    # data
    batch_size: int = 8
    # adamw optimizer
    train_steps: int = 500000 # total number of training iterations
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    learning_rate: Union[float, CosineDecayScheduleConfig] = field(default_factory=CosineDecayScheduleConfig)
    # wandb logging
    wandb: Optional[WandbConfig] = field(default_factory=WandbConfig)
    # model
    model: GPTConfig = field(default_factory=GPTConfig)


@partial(jax.pmap, axis_name='batch')
def train_step(state: TrainState, tokens: jnp.ndarray, dropout_key) -> Tuple[jnp.ndarray, TrainState]:

    dropout_key = jax.random.fold_in(dropout_key, state.step)

    def loss_fn(params: FrozenDict) -> jnp.ndarray:
        X, Y = tokens[:, :-1], tokens[:, 1:]
        logits = state.apply_fn(params, X, False, rngs={'dropout': dropout_key})
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
def eval_step(state: TrainState, tokens: jnp.ndarray) -> jnp.ndarray:
    X, Y = tokens[:, :-1], tokens[:, 1:]
    logits = state.apply_fn(state.params, X, True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return loss


def evaluate(state: TrainState, data: np.memmap, batch_size: int, block_size: int, steps: int) -> jnp.ndarray:
    n = jax.local_device_count() * config.batch_size * (block_size + 1)
    data = data[:(len(data) // n) * n].reshape(-1, jax.local_device_count(), batch_size, block_size + 1)
    losses = []

    for tokens in data[:steps]:
        loss = eval_step(state, tokens)
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


def init_train_state(key, config: GPTConfig, optimizer, weight_decay=1e-2) -> TrainState:

    model = GPT(config)

    params = model.init(key)

    optimizer = optax.chain(
        # Apply weight decay only to non-bias parameters
        optax.add_decayed_weights(weight_decay, mask=param_decay_mask(params)),
        optimizer,
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)

    return train_state


def get_train_batch(data: np.memmap, batch_size: int, block_size: int, key):
    with jax.experimental.enable_x64():
        ix = jax.random.randint(key, [jax.local_device_count() * batch_size], 0, len(data) - block_size)
        ix = np.asarray(ix)
        tokens = np.stack([data[i:i+block_size+1] for i in ix])
    return tokens.reshape(-1, batch_size, block_size+1)


def maybe_download(path: str) -> str:
    assert gfile.exists(path), f'file {path} does not exist'
    if path.startswith('gs://'):
        local_path = Path(path.split('/')[3:])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        gfile.copy(path, str(local_path))
        return local_path
    return path


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb.config = asdict(config)

    block_size = config.model.block_size

    # ===== datasets =====
    # each process downloads it's own shard
    train_path = gfile.join(config.data_dir, f'train_{jax.process_index()}.bin')
    val_path = gfile.join(config.data_dir, f'val_{jax.process_index()}.bin')
    train_path = maybe_download(train_path)
    val_path = maybe_download(val_path)

    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    # ===== learning rate schedule =====
    if isinstance(config.learning_rate, CosineDecayScheduleConfig):
        learning_rate = optax.warmup_cosine_decay_schedule(**asdict(config.learning_rate))
    else:
        learning_rate = config.learning_rate

    # =====  init parameters ============
    key = jax.random.PRNGKey(config.seed)
    key, key_params, key_dropout = jax.random.split(key, 3)
    keys_dropout = jax.random.split(key_dropout, jax.device_count())
    # make sure dropout keys are different for each device (local and global)
    keys_dropout = jnp.array_split(keys_dropout, jax.process_count())[jax.process_index()]

    optimizer = optax.adam(learning_rate, *config.betas)

    train_state = init_train_state(key_params, config.model, optimizer, config.weight_decay)

    num_params = count_params(train_state.params)
    if jax.process_index() == 0:
        #logging.info(f'PARAMETER COUNT: {num_params:,}')
        print(f'PARAMETER COUNT: {num_params:,}')

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

    for step in range(step, config.train_steps):

        if step % config.eval_interval == 0:
            val_loss = evaluate(train_state, val_data, config.batch_size,
                                block_size, config.eval_steps)
            
            if (val_loss < best_val_loss) and not config.eval_only:
                best_val_loss = val_loss
                # save train state in process 0
                checkpoints.save_checkpoint_multiprocess(
                    f'{config.out_dir}/checkpoints/train_state',
                    (unreplicate(train_state), key), step, keep=3)
                
            if (config.wandb is not None) and (jax.process_index() == 0):
                wandb.log({"val/loss": val_loss}, step=step)

        if config.eval_only:
            break

        key, key_batch = jax.random.split(key)
        tokens = get_train_batch(train_data, config.batch_size, block_size, key_batch)
        loss, train_state = train_step(train_state, tokens, keys_dropout)

        if (config.wandb is not None) and (jax.process_index() == 0):
            wandb.log({
                "train/loss": loss,
                "lr": learning_rate(step) if callable(learning_rate) else learning_rate
            }, step=step)