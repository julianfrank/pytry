from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()


network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
