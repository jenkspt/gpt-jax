from typing import Optional
import jax
import tensorflow as tf


OPTIONS = tf.data.Options()
OPTIONS.deterministic = True
OPTIONS.autotune.enabled = True


def get_dataset(pattern: str,
                batch_size: int = 8,
                block_size: int = 1024,
                shuffle_buffer_size: Optional[int] = None,
                repeat: Optional[int]=None,
                seed: Optional[int]=None) -> tf.data.Dataset:

    tf.random.set_seed(seed)

    file_ds = tf.data.Dataset.list_files(pattern, shuffle=bool(shuffle_buffer_size))
    file_ds = file_ds.shard(jax.process_count(), jax.process_index())
    ds = tf.data.TFRecordDataset(file_ds, num_parallel_reads=tf.data.AUTOTUNE)
    # each element of the dataset is a tokenized string
    feature_description = {
        'ids': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    def parse_example(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.decode_raw(example['ids'], tf.uint16)

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat(repeat)

    # here we shuffle each group of tokens and then unbatch into a single
    # contiguous sequence of ids, we then chunk the sequence into blocks
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.unbatch().batch(block_size + 1, drop_remainder=True)

    # each block is then shuffled and then batched
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.batch(jax.local_device_count(), drop_remainder=True)
    ds = ds.with_options(OPTIONS)
    return ds.prefetch(2)
