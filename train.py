from typing import Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from functools import partial
import os
import wandb
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
from dataset import get_dataset

import tensorflow as tf

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass(frozen=True)
class WandbConfig:
    """
    wandb logging configuration
    """
    entity: str = 'jenkspt'
    """username or team name where you're sending runs"""
    project: str = 'owt'
    """project name"""
    name: str = 'test'
    """experiment name"""
    mode: str = 'online'
    """'offline', 'online', or 'disabled'"""
    notes: str = ''


@dataclass(frozen=True)
class CosineDecayScheduleConfig:
    init_value: float = 0.0
    peak_value: float = 2.5e-4
    warmup_steps: int = 2000
    decay_steps: int = 150000
    end_value: float = 1e-5


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 555
    out_dir: str = 'out'                        # output directory for checkpoints (can be gcs path)
    train_pattern: str = 'train_??.tfrecord'    # training files glob pattern (can be gcs path)
    val_pattern: str = 'val_??.tfrecord'        # validation files glob pattern (can be gcs path)
    shuffle_buffer_size: int = 128
    eval_interval: int = 500
    eval_steps: int = 16        # evaluate for this number of steps (per-device)
    eval_only: bool = False     # if True, script exits right after the first eval
    keep_checkpoints: int = 3   # number of historical checkpoints to keep
    batch_size: int = 16        # per-device batch size
    train_steps: int = 150000   # total number of training iterations
    weight_decay: float = 1e-2  # not applied to bias and embedding parameters
    grad_clip: float = 1.0      # gradient norm clipping magnitude
    gradient_accumulation_steps: int = 1    # used to simulate larger batch sizes
    betas: Tuple[float, float] = (0.9, 0.95) # adamw optimizer betas
    learning_rate: CosineDecayScheduleConfig = field(default_factory=CosineDecayScheduleConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig) # wandb logging
    model: GPTConfig = field(default_factory=GPTConfig)     # gpt model config
    remat: bool = False    # set to True to rematerialize gradients during backward pass


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


def evaluate(state: TrainState, ds: tf.data.Dataset, batch_size: int, block_size: int, steps: int) -> jnp.ndarray:
    losses = []
    for _, tokens in zip(range(steps), ds):
        tokens = tokens._numpy()
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


def init_train_state(key, config: TrainConfig, learning_rate) -> TrainState:

    if config.remat:
        model = flax.linen.remat(GPT,
            static_argnums=(2,),
            policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)(config.model)
    else:
        model = GPT(config.model)

    params = model.init(key)

    optimizer = optax.chain(
        # Apply weight decay only to non-bias parameters
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(learning_rate, *config.betas, weight_decay=config.weight_decay, mask=param_decay_mask(params)),
        optax.apply_every(config.gradient_accumulation_steps),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)

    return train_state


def get_default_config() -> TrainConfig:
    # use this file to set default values
    path = os.environ.get('GPT_CONFIG', os.path.join('config', 'gpt2.yaml'))
    if not os.path.exists(path):
        return TrainConfig()
    logging.info(f'using config file at {path}')
    with open(path, 'r') as f:
        return tyro.from_yaml(TrainConfig, f)


if __name__ == "__main__":
    config = tyro.cli(TrainConfig, default=get_default_config())

    if config.wandb is not None and jax.process_index() == 0:
        wandb.init(**asdict(config.wandb))
        wandb.config.update(asdict(config))

    block_size = config.model.block_size

    # ===== datasets =====
    train_ds = get_dataset(
        config.train_pattern, config.batch_size,
        block_size, config.shuffle_buffer_size,
        seed = config.seed)

    val_ds = get_dataset(
        config.val_pattern, config.batch_size,
        block_size, repeat=1)

    # =====  init parameters ============
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

    best_val_loss = float('inf')

    # ==== restore dataset and train state ==== #
    # restore unreplicated optimizer + model state from last checkpoint.
    # this is a no-op if no checkpoints exist
    train_state = checkpoints.restore_checkpoint(
        f'{config.out_dir}/checkpoints/train_state', train_state)

    # grab step from last checkpoint
    step = int(train_state.step)


    train_iter = iter(train_ds)
    # We need to be able to save the dataset state for stopping and resuming training
    # we'll save a dataset checkpoint for each shard
    dataset_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(iterator=train_iter),
        f"{config.out_dir}/checkpoints/dataset_{jax.process_index()}",
        max_to_keep=config.keep_checkpoints)
    dataset_manager.restore_or_initialize()

    # replicate parameters to each device
    train_state = replicate(train_state)

    for step in range(step, config.train_steps):

        if step % config.eval_interval == 0:
            val_loss = evaluate(train_state, val_ds, config.batch_size,
                                block_size, config.eval_steps)

            if config.eval_only:
                break

            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                if jax.process_index() == 0:
                    # save train state in process 0
                    checkpoints.save_checkpoint(
                        f'{config.out_dir}/checkpoints/train_state',
                        unreplicate(train_state), step, keep=config.keep_checkpoints, overwrite=True)
                dataset_manager.save(step)
                
            if (config.wandb is not None) and (jax.process_index() == 0):
                wandb.log({"val/loss": val_loss}, step=step)

        tokens = next(train_iter)._numpy()
        loss, train_state = train_step(train_state, tokens, keys_dropout)

        if (config.wandb is not None) and (jax.process_index() == 0):
            wandb.log({
                "train/loss": loss[0].item(),
                "lr": learning_rate(step) if callable(learning_rate) else learning_rate,
                "step": step,
                "block": step * config.batch_size * jax.device_count(),
            }, step=step)