import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.datasets import fashion_mnist

#Data preprocessing
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train/255.0
X_test = X_test/255.0

#Reshaping the data from 28*28 to 784*1
#-1 for all the images till the last element
X_train = X_train.reshape(-1, 28*28)

X_test = X_test.reshape(-1, 28*28)

#Building ANN

model = tf.keras.models.Sequential()

#Adding first fully connected hidden layer (Dense class)
model.add(tf.keras.layers.Dense(units=128, activation="relu", input_shape=(784, )))

#Adding a second layer with dropout dropping 20% neurons
model.add(tf.keras.layers.Dropout(0.2))

#Adding output layer
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

#COMPILING THE MODEL
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss: ",test_loss)
print("Accuracy on test: ",test_accuracy)
