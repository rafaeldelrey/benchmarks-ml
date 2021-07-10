"""
Lecun CNN architecture. MNIST benchmark
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

"""

import time
from datetime import timedelta
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.framework.ops import disable_eager_execution
import wandb

from utils import setup_optimizations, is_using_gpu


disable_eager_execution()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def main():
    # tf.enable_v2_behavior()

    # from tensorflow.python.compiler.mlcompute import mlcompute
    # mlcompute.set_mlc_device(device_name='gpu')

    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    params = dict(
        model="mnist_lecun",
        group="group_001",
        using_gpu=is_using_gpu(),
        use_mixed_precision=False,
        use_xla=False,
    )

    setup_optimizations(params)

    wandb.init(project="benchmarks-ml",
               entity="rafaeldelrey",
               config=params)

    ti = time.time()

    batch_size = 128

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"],
    )

    model.fit(
        ds_train,
        epochs=5,
        validation_data=ds_test,
        # callbacks=[wandb.keras.WandbCallback(save_model=False)],
    )

    td = time.time() - ti
    print(f"Elapsed: {timedelta(seconds=td)}")
    wandb.log(dict(elapsed=td))
    wandb.join()


if __name__ == "__main__":
    main()
