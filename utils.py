import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def setup_optimizations(params):
    if params.get("use_xla", False):
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    if params.get("use_mixed_precision", False):
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)


def is_using_gpu():
    return tf.test.is_gpu_available()