import tensorflow as tf
import tensorflow_datasets as tfds

def _preprocess(example):
    img = tf.cast(example["image"], tf.float32) / 255.0
    # Keep images in [0, 1] for flow matching
    # Flow matching/MeanFlow assumes data is in [0, 1] when using Gaussian noise
    label = tf.cast(example["label"], tf.int32)
    return img, label

def make_cifar10(batch_size: int, split: str, shuffle: bool=True, cache: bool=True, filter_classes=None):
    ds = tfds.load("cifar10", split=split, as_supervised=False)
    # IMPORTANT: Map BEFORE cache/shuffle to ensure correct preprocessing
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter to specific classes if requested
    if filter_classes is not None:
        def filter_fn(img, label):
            return tf.reduce_any(tf.equal(label, filter_classes))
        ds = ds.filter(filter_fn)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
