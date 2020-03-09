import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#using mnist data set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9

#this portion displays the picture as a pyplot!
#plt.imshow(x_test[0], cmap=plt.cm.binary)
#plt.show()

#reshape images
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=3)

model.save('num_reader.model')

new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))
