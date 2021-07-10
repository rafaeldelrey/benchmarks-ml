import time
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import wandb

from utils import setup_optimizations, is_using_gpu


def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    params = dict(
        model="cifar10",
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

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels)
    )

    td = time.time() - ti
    print(f"Elapsed: {timedelta(seconds=td)}")
    wandb.log(dict(elapsed=td))
    wandb.join()


if __name__ == "__main__":
    main()
