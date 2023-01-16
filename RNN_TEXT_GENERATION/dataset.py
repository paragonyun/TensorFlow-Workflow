import tensorflow as tf

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text)) # text 내부의 문자를 모두 알아냅니다.

"""
1. 
tf.strings.unicode_split : text를 토큰으로 분할시켜줍니다.
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
>>> <tf.RaggedTensor [[b'a', b'b', b'c', b'd', b'e', b'f', b'g'], [b'x', b'y', b'z']]>

2. 
tf.keras.layers.StringLookup : 문자-숫자 indexing이 어떻게 되어 있는지를 반환합니다.
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
ids = ids_from_chars(chars)
>>> <tf.RaggedTensor [[40, 41, 42, 43, 44, 45, 46], [63, 64, 65]]>

2-1.
** StringLookup의 .get_vocabulary()는 ids가 들어오면 char가 뭔지 알려줍니다.
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    >>> <tf.RaggedTensor [[b'a', b'b', b'c', b'd', b'e', b'f', b'g'], [b'x', b'y', b'z']]>

3. 
tf.strings.reduce_join() : 문자들을 문자열로 결합합니다.
tf.strings.reduce_join(chars, axis=-1).numpy()
>>> array([b'abcdefg', b'xyz'], dtype=object)
"""

# id - char
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
# char-id (invert=True)!!
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

def text_from_ids(ids):
    """id가 들어오면 그걸 문자로 바꾸고 그 문자를 문자열로 바꾸는 함수입니다."""
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_len = 100
seqeunces = ids_dataset.batch(seq_len+1, drop_remainder=True)

def split_input_target(seq):
    input_text = seq[:-1]
    target_text = seq[1:]
    return input_text, target_text

dataset = seqeunces.map(split_input_target)
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

def get_dataset():
    print("Making Dataset is Successful")
    return dataset












