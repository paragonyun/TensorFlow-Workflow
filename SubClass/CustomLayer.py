import tensorflow as tf
from tensorflow import keras

"""
가중치 없이 단순 값 변경을 위한 Layer는 Lambda로 하는 게 편함
"""
exp_layer = keras.layers.Lambda(lambda x: tf.exp(x))

"""
커스텀 Dense Layer
"""


class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(
        self, batch_input_shape
    ):  # torch는 그냥 init에서 모델을 정의하지만 여기선 build에서 말 그래도 쌓는다!
        self.kernel = self.add_weight(
            name="kernel",
            shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal",
        )
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros"
        )

        """마지막에 꼭 부모클래스의 build()를 호출해야 build함수를 짰다고 인식할 수 있습니다."""
        super().build(batch_input_shape)

    def call(self, x):
        return self.activation(
            x @ self.kernel + self.bias
        )  # 행렬곱 + bias 연산 후 activation

    def compute_output_shape(self, batch_input_shape):
        print(batch_input_shape.as_list())
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
        }


dense = MyDense(units=10, activation="relu")

"""
Input을 여러개로 받아야 하는 경우, 튜플로 묶고 call 메서드 안에서 언패킹 해주면 된다.

다만 list로 반환해야 합니다.
"""


class MultipleInput(keras.layers.Layer):
    def call(self, tupled_X):
        x1, x2 = tupled_X
        return [x1 + x2, x1 * x2, x1 / x2]

    def compute_output_shape(self, batch_input_shape):
        b1, b2, b3 = batch_input_shape
        return [b1, b2, b3]


"""
dropout처럼 훈련때만 사용하고 test 시에는 사용하지 않는 기능을 넣고 싶으면
trainable을 놓으면 된다.

어차피 super.__init__(**kwargs)로 상속 받아서 사용 가능!
"""


class OnlyWhenTraining(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, x, trainable=None):
        if trainable:
            noise = tf.random.normal(tf.shape(x), stddev=self.stddev)
            return x + noise
        else:
            return x

    def return_ouptut_shape(self, batch_input_shape):
        return batch_input_shape
