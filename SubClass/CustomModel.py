"""Residual Block을 한번 만들어볼 거임"""
import tensorflow as tf
from tensorflow import keras


class MyResidualLayer(keras.layers.Layer):
    """일단 Layer를 생성"""

    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            keras.layers.Dense(
                n_neurons, activation="elu", kernel_initializer="he_normal"
            )
            for _ in range(n_layers)
        ]  ## 이런식으로 n_layer 겹의 Dense를 만들 수 있음

    def call(self, x):
        z = x
        for layer in self.hidden:
            z = layer(z)
        return x + z


class MyResidualModel(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(
            30, activation="elu", kernel_initializer="he_normal"
        )

        self.res_block1 = MyResidualLayer(n_layers=2, n_neurons=30)
        self.res_block2 = MyResidualLayer(n_layers=2, n_neurons=30)

        self.out = keras.layers.Dense(output_dim)

    def call(self, x):
        z = self.hidden1(x)
        for _ in range(3):
            """res_block을 3번 돌고 나온 결과를 활용할 예정"""
            z = self.res_block1(z)
        z = self.res_block2(z)
        return self.out(z)
