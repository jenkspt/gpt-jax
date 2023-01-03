from typing import Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from functools import partial
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
import tensorflow as tf

from model import GPT, GPTConfig


@dataclass(frozen=True)
class WandBConfig:
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
    wandb: Optional[WandBConfig] = field(default_factory=WandBConfig)
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


def evaluate(state: TrainState, dataset, key) -> jnp.ndarray:

    # we want a different dropout rng key for each device
    keys = jax.random.split(key, jax.device_count())
    keys_dropout = jnp.array_split(keys, jax.process_count())[jax.process_index()]
    losses = []
    for tokens in dataset:
        tokens = tokens._numpy().reshape(-1, config.batch_size, block_size)
        loss = eval_step(state, tokens, {'dropout': keys_dropout})
        losses.append(loss)
    return jnp.mean(jnp.stack(losses))


def count_params(params: FrozenDict) -> int:
    p = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, p)


def not_bias(params: FrozenDict) -> FrozenDict:
    """ pytree mask for non-bias parameters """
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_param_mask = {k: not k[-1].endswith('bias') for k in flat_params.keys()}
    param_mask = flax.traverse_util.unflatten_dict(flat_param_mask)
    return frozen_dict.freeze(param_mask)


def get_dataset(path, batch_size, block_size, shuffle_buffer_size=0, repeat=1, seed=None):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.repeat(repeat)
    ds = ds.batch(block_size, drop_remainder=True)
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(2)


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    if config.wandb is not None:
        wandb.init(**asdict(config.wandb))
        wandb.config = asdict(config)

    block_size = config.model.block_size

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
        optax.add_decayed_weights(config.weight_decay, mask=not_bias(params)),
        optax.adamw(learning_rate, *config.betas, weight_decay=0.0),
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
    del params

    val_ds = get_dataset('/home/penn/drive2/val.bin', config.batch_size, block_size)
    train_ds = get_dataset(
        '/home/penn/drive2/train.bin',
        config.batch_size, block_size,
        config.shuffle_buffer_size, repeat=None, seed=config.seed) 
    train_iter = iter(train_ds)

    # manage dataset state (each process saves it's own state)
    dataset_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(iterator=train_iter),
        f'{config.out_dir}/checkpoints/dataset_{jax.process_index()}',
        max_to_keep=3)

    step = 0
    best_val_loss = float('inf')

    # ==== restore dataset and train state ==== #
    # restore unreplicated optimizer + model state from last checkpoint.
    # this is a no-op if no checkpoints exist
    train_state = checkpoints.restore_checkpoint(
        f'{config.out_dir}/checkpoints/train_state', train_state)

    # grab step from last checkpoint
    step = int(train_state.step)
    dataset_manager.restore_or_initialize()

    # replicate parameters to each device
    train_state = replicate(train_state)

    for step, tokens in enumerate(train_iter, start=step):
        tokens = tokens._numpy().reshape(-1, config.batch_size, block_size)

        if step % config.eval_interval == 0:
            key, key_eval = jax.random.split(key)
            val_loss = evaluate(train_state, val_ds, key_eval)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                dataset_manager.save(step)  # save dataset state in each process
                # save train state in process 0
                checkpoints.save_checkpoint_multiprocess(
                    f'{config.out_dir}/checkpoints/train_state',
                    unreplicate(train_state), step, keep=3)
                
                if (config.wandb is not None) and (jax.process_index() == 0):
                    wandb.log({"val/loss": val_loss}, step=step)

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