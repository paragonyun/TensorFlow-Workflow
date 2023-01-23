import tensorflow as tf
from tensorflow import keras


class MyHuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)  # 예전에 만들었던 함수
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """이 class를 함수처럼 사용할 때 사용, 변수를 업데이트 한다."""
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count  # 현재 점수 리턴

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
