import time
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import wandb

from utils import setup_optimizations, is_using_gpu


# ONLY ON THE MAC
# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='gpu')


def main():
    print("\nMNIST dataset")
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    using_gpu = tf.test.is_gpu_available()

    params = dict(
        model="mnist",
        group="group_002",
        using_gpu=is_using_gpu(),
        use_mixed_precision=False,
        use_xla=False,
    )

    setup_optimizations(params)

    wandb.init(project="benchmarks-ml",
               entity="rafaeldelrey",
               config=params)

    ti = time.time()
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    td = time.time() - ti
    print(f"Elapsed: {timedelta(seconds=td)}")

    wandb.log(dict(elapsed=td))
    wandb.join()


if __name__ == "__main__":
    main()
