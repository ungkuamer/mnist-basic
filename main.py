import tensorflow as tf
import numpy as np

# data loading 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalisation of data (range only from 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # flatten the data set 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 neurons and Rectified Linear Unit as activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # second layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # 10 neurons and Softmax as activation function (prob. distribution)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] ) # models always try to minimise loss not improve accuracy
model.fit(x_train, y_train, epochs=1) # run the training process 3 times

val_loss, val_acc = model.evaluate(x_test, y_test)

model.save('num_reader_basic.model')

new_model = tf.keras.models.load_model('num_reader_basic.model')
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))


