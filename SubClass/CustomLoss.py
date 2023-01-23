"""
keras.losses.Loss를 상속받아 진행
"""
import tensorflow as tf
import keras


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < 1
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        """
        하이퍼파라미터 이름과 같이 매핑된 딕셔너리 반환
        -> 부모 클래스의 get_config를 호출하고 그 다음에 새로 threshold라는 파라미터를 추가함
        """
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
