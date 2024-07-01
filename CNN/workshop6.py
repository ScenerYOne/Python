import numpy as np
from keras.datasets.mnist import load_data
from keras.utils import plot_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import tensorflow as tf
import time

# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Reshape data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Define input shape and number of classes
in_shape = (224, 224, 3)
num_classes = 10

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=in_shape))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Plot model
plot_model(model, 'digit.png', show_shapes=True, show_layer_names=True)

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resize and train data
x_train_reshaped = tf.reshape(x_train, [-1, 28, 28, 1])
x_train_resized = tf.image.resize(x_train_reshaped, [224, 224])

# Train model
start = time.time()
history = model.fit(x_train_resized, y_train, epochs=15, batch_size=128, verbose=1, validation_split=0.1)
end = time.time()
print("Time Taken: {:.2f} minutes".format((end - start) / 60))
