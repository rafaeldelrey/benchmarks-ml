"""
https://www.tensorflow.org/text/tutorials/text_classification_rnn

"""
import time
from datetime import timedelta
import tensorflow_datasets as tfds
import tensorflow as tf
import wandb

from utils import setup_optimizations, is_using_gpu


def main():
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    params = dict(
        model="text_rnn",
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

    buffer_size = 10000
    batch_size = 64

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    vocab_size = 1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    model.fit(train_dataset,
              epochs=2,
              validation_data=test_dataset,
              validation_steps=30)

    td = time.time() - ti
    print(f"Elapsed: {timedelta(seconds=td)}")
    wandb.log(dict(elapsed=td))
    wandb.join()


if __name__ == "__main__":
    main()
