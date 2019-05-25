from tensorflow.python.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

from tensorflow.python.keras import models
from tensorflow.python.keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
