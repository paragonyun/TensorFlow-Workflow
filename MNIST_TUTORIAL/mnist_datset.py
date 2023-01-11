

import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

## 노란줄 뜨긴 하는데 되긴 됨
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def mnist_dataset():
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, X_test = X_train / 255.0, X_test / 255.0

    # print(X_train[0].shape) # 28, 28

    ## 기본적으로 가져오는 shape이 (28,28)이기 때문에 Channel을 추가해줄 거임
    ## tf.newaxis : 해당 위치에 axis를 추가해준다. torch의 unsqueeze랑 같은 역할!
    X_train = X_train[..., tf.newaxis].astype('float32')
    X_test = X_test[..., tf.newaxis].astype('float32')

    # print(X_train[0].shape) # (28, 28, 1) 

    ## tf.data로 배치 생성(Dataloader 생성)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return train_ds, test_ds



