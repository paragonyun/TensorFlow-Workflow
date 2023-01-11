import tensorflow as tf
from tqdm import tqdm

from mnist_datset import mnist_dataset
from model import Model
from train import Trainer

loss_fn = tf.keras.losses\
            .SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics\
                .SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics\
                .SparseCategoricalAccuracy(name='test_accuracy')

train_data, test_data = mnist_dataset()

model = Model()

loop = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loss=train_loss,
    train_acc=train_acc,
    test_loss=test_loss,
    test_acc=test_acc)

EPOCHS = 10

print("🚩Model Architecture!🚩")
model.build(input_shape=(32,28,28,1))
print(model.summary())

print("🚀Start Training...🚀")

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    # training
    for imgs, labels in tqdm(train_data):
        loop.train(imgs, labels)

    # inference
    for imgs, labels in tqdm(test_data):
        loop.test(imgs, labels)

    print(
        f'Epoch {epoch + 1}\n'
        f'Loss: {train_loss.result()}\t'
        f'Accuracy: {train_acc.result() * 100}\n'
        f'Test Loss: {test_loss.result()}\t'
        f'Test Accuracy: {test_acc.result() * 100}\n'
    )
