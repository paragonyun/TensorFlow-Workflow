import tensorflow as tf
"""https://www.tensorflow.org/text/tutorials/text_generation"""
class MyModel(tf.keras.Model):
    def __init__(self, 
                vocab_size: int = 65, 
                embedding_dim: int = 256, 
                rnn_units: int = 1024):
        super(MyModel, self).__init__()

        # 각 문자를 embedding_dim 만큼의 차원으로 매핑합니다.
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim
        )
        self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self,
            inputs,
            states=None,
            return_state=False,
            training=False):
        
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x) # 오~ 이렇게 초기 hidden_state를 설정 안 해줘도 된다!

        x, states = self.gru(x, initial_state=states, training=training)
        output = self.dense(x, training=training)

        if return_state:
            return output, states
        
        else:
            return output
        
