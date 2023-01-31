import tensorflow as tf


"""from_tensor_slices
텐서를 받아 차원을 따라 각 원소가 item으로 표현되게 만듦
생성된 데이터 -> (10, ) 
"""
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)

print("tf.tensor_from_slices()로 만든 dataset iteration")
for item in dataset:
    print(item)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # tf.Tensor(2, shape=(), dtype=int32)
    # tf.Tensor(3, shape=(), dtype=int32)
    # tf.Tensor(4, shape=(), dtype=int32)
    # tf.Tensor(5, shape=(), dtype=int32)
    # tf.Tensor(6, shape=(), dtype=int32)
    # tf.Tensor(7, shape=(), dtype=int32)
    # tf.Tensor(8, shape=(), dtype=int32)
    # tf.Tensor(9, shape=(), dtype=int32)

"""repaet()
원본 데이터셋의 아이템을 반복함
데이터셋의 메서드는 새로운 데이터셋을 return 하므로 항상 반환 객체를 정해야 함!
"""
print("\n\nrepeat()로 3번 반복해서 만든 예시")
repeated = dataset.repeat(5)
for item in repeated:
    print(item)
    # 0~9가 3번 반복해서 나옴

"""batch()
dataset의 item을 batch개씩 묶어서 반환
n개씩 묶고 남는 건 모자란 대로 그냥 return 하는데, drop_remainder=True로 하면 그런 애들 다 버림
"""
print("\n\nbatch()로 7개씩 묶기 - default")
batch_daetaset = repeated.batch(7)
for item in batch_daetaset:
    print(item)
    # `tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
    # tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
    # tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
    # tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
    # tf.Tensor([8 9 0 1 2 3 4], shape=(7,), dtype=int32)
    # tf.Tensor([5 6 7 8 9 0 1], shape=(7,), dtype=int32)
    # tf.Tensor([2 3 4 5 6 7 8], shape=(7,), dtype=int32)
    # tf.Tensor([9], shape=(1,), dtype=int32)`
print("\nbatch()로 7개씩 묶기 - drop_remainder=True")
batch_daetaset = repeated.batch(7, drop_remainder=True)
for item in batch_daetaset:
    print(item)
    # tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
    # tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
    # tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
    # tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
    # tf.Tensor([8 9 0 1 2 3 4], shape=(7,), dtype=int32)
    # tf.Tensor([5 6 7 8 9 0 1], shape=(7,), dtype=int32)
    # tf.Tensor([2 3 4 5 6 7 8], shape=(7,), dtype=int32)

"""shuffle()
원리 : 카드에서 n개 뽑고, 그 n개중 하나 뽑아서 새로운 댁을 만드는 데에 씀
그리고 n-1개 됐으니까 원래 카드에서 또 1개 뽑아서 위의 과정 반복
=> n개를 쌓는 수가 shuffle을 잘 하는 데에 중요하다! 그 n을 buffer_size라고 한다.
=> 완벽하게 하려면 buffer수를 dataset의 수와 같게 하는 게 좋음
"""
print("\n\nshuffle()로 섞은 데이터셋")
shuffled = repeated.shuffle(buffer_size=5, seed=42).batch(7)
for item in shuffled:
    print(item)
    # tf.Tensor([0 2 3 6 7 9 4], shape=(7,), dtype=int32)
    # tf.Tensor([5 0 1 1 8 6 5], shape=(7,), dtype=int32)
    # tf.Tensor([4 8 7 1 2 3 0], shape=(7,), dtype=int32)
    # tf.Tensor([5 4 2 7 8 9 1], shape=(7,), dtype=int32)
    # tf.Tensor([3 6 0 4 6 3 9], shape=(7,), dtype=int32)
    # tf.Tensor([9 2 7 5 3 1 5], shape=(7,), dtype=int32)
    # tf.Tensor([6 0 2 7 8 8 9], shape=(7,), dtype=int32)
    # tf.Tensor([4], shape=(1,), dtype=int32)
