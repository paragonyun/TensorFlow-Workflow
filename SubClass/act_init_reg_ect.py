"""
활성화 함수, 초기화, 규제, 제한도 커스터마이즈 할 수 있다.
"""
import tensorflow as tf


def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2.0 / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01, *weights))


def my_positive_weights(weights):  # 렐루
    return tf.where(weights < 0.0, tf.zeros_like(weights), weights)


## 적용
layer = tf.keras.layers.Dense(
    30,
    activation=my_softplus,
    kernel_initializer=my_glorot_initializer,
    kernel_regularizer=my_l1_regularizer,
    kernel_constraint=my_positive_weights,
)

## Loss와 마찬가지로 상속이 필요한 경우 class화
from tensorflow import keras


class MyL1Regularizer(keras.regularizers.Regularizer):
    """주의할 점
    손실, Layer, 모델은 call() 이지만
    규제, 초기화, 제한은 __call__()을 작성해야 한다.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}
