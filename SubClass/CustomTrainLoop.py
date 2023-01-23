import tensorflow as tf
from tensorflow import keras

import numpy as np

# 간단한 모델
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential(
    [
        keras.layers.Dense(
            30,
            activation="elu",
            kernel_initializer="he_normal",
            kernel_regularizer=l2_reg,
        ),
        keras.layers.Dense(1, kernel_regularizer=l2_reg),
    ]
)

# 훈련 데이터를 랜덤하게 추출 (중복은 허용)
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


# print 함수 (tqdm 써도 됨 사실)
def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(
        [f"{m.name} : {m.result():.4f}" for m in [loss] + (metrics or [])]
    )
    end = "" if iteration < total else "\n\n"
    print(f"\r[{iteration}/{total}]\t" + metrics, end=end)


X_train = np.random.randint(5000, size=5000)
X_train = X_train.reshape(-1, 1)
y = np.random.randint(5000, size=5000)
print(len(X_train), len(y))

EPOCHS = 5
BATCH_SIZE = 32
N_STEPS = len(X_train) // BATCH_SIZE

optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]

"""여기부터 Train Loop"""
for epoch in range(1, EPOCHS + 1):
    print(f"\nEPOCH {epoch}/{EPOCHS}")

    for step in range(1, N_STEPS + 1):
        X_batch, y_batch = random_batch(X_train, y)

        with tf.GradientTape() as tape:
            y_pred = model(X_batch, y_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))  ## Batch 별 평균 loss 계산
            loss = tf.add_n(
                [main_loss] + model.losses
            )  # Elemental Wise Sum으로 모든 loss를 더해줌

        gradients = tape.gradient(
            loss, model.trainable_variables
        )  # 훈련 가능 변수들에 대한 gradients를 계산함
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )  # 훔련 가능 변수들에 대한 gradient를 optimizer를 통해 적용 (torch -> optimizer.step())
        mean_loss(loss)  # 평균 loss 계산

        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * BATCH_SIZE, len(y), mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()  # metric 초기화 (torch.zero_grad()를 여기서 하는듯?)
