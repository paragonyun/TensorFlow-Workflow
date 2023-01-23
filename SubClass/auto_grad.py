import tensorflow as tf
from tensorflow import keras


def f(w1, w2):
    return 3 * w1**2 + 2 * w1 * w2


w1, w2 = tf.Variable(5.0), tf.Variable(3.0)

"""tf.GradientTape로 감싸주면 해당 변수와 관련된 모든 연산을 자동으로 기록
각 미분의 결과가 36, 10으로 잘 기록되었음을 확인할 수 있음.
"""

with tf.GradientTape() as tape:
    z = f(w1, w2)

"""tape.gradient를 호출하면 tape에 기록 되어 있던 gradient들이 다 지워짐! 주의할것!
계속 유지하고 싶으면
with tf.GradientTape(persistent=True) as tape 로 사용
"""
grads = tape.gradient(z, [w1, w2])
print(
    grads
)  # [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>, <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
