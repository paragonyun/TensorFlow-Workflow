import tensorflow as tf

class Trainer:
    def __init__(self, 
                model, 
                loss_fn, 
                optimizer, 
                train_loss, 
                train_acc,
                test_loss, 
                test_acc):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.test_loss= test_loss
        self.test_acc = test_acc

    @tf.function
    def train(self, imgs, labels):
        """
        tensorflow는 @tf.function을 붙이면 함수 내의 로직이 
        그래프의 생성과 실행을 분리시키기 때문에 속도에 조금 더 이점이 있습니다.
        그러나 그런 만큼 과정을 살펴볼 수가 없기 때문에 
        처음 개발할 땐 @tf.function을 안 붙였다가 이후에 실행만 할 때 붙이는 걸 추천합니다.
        """
        with tf.GradientTape() as tape:
            """
            tensorflow는 자동미분을 위해 tf.GradientTape를 지원합니다.
            이름 그대로 with context 안에서 이루어진 모든 연산을 tape에 저장합니다.
            이후 Gradient를 계산할 때 tape에 저장된 연산을 이용합니다.
            """
            preds = self.model(imgs, training=True)
            loss = self.loss_fn(labels, preds)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_acc(labels, preds)

    @tf.function
    def test(self, imgs, labels):
        """
        test를 할 땐 gradient가 필요 없으므로 그냥 합니다.
        """
        preds = self.model(imgs, training=False)
        test_loss = self.loss_fn(labels, preds)

        self.test_loss(test_loss)
        self.test_acc(labels, preds)
        
    
