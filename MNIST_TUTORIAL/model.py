from keras.layers import Dense, Flatten, Conv2D
from keras import Model

class Model(Model):
    def __init__(self, ):
        super(Model, self).__init__()

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10)

    ## tensorflow는 call 함수로 torch의 forward 역할을 적는다.
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out

