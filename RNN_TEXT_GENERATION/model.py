import tensorflow as tf
"""https://www.tensorflow.org/text/tutorials/text_generation"""
class MyModel(tf.keras.Model):
    def __init__(self, 
                vocab_size: int = 66, 
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
        

class Generator(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature):
        super(Generator, self).__init__()

        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # MASK를 생성합니다.
        skip_ids = self.ids_from_chars(['UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
                    values=[-float('inf')] * len(skip_ids), # ids 수만큼 -inf 생성
                    indices=skip_ids,
                    dense_shape=[len(ids_from_chars.get_vocabulary())]
        )

        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, 
                                                states=states,
                                                return_state=True)
        
        # 마지막 예측값만 사용
        predicted_logits = predicted_logits[:, -1, :] # 자동으로 list에 넣어주는듯
        predicted_logits = predicted_logits/self.temperature

        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states