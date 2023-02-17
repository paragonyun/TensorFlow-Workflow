from dataset import *
from model import *

import os

dataset = get_dataset()

model = MyModel()

# 모델이 Logit을 반환하면 from_logits=True로 지정해줘야 합니다.
# Soft max를 마지막에 안 취해줘서 그런듯?
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "chpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS = 20

hist = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
