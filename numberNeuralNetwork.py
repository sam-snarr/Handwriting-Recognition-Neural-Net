# Sam Snarr 
# 3/23/19

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# dataset of handwritten digits 0-9
mnist = tf.keras.datasets.mnist

# splits dataset up into train and test
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
dat = xTest

# normalize data
xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)

# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=1)

# output heuristics
validationLoss, validationAccuracy = model.evaluate(xTest, yTest)

print('\nval loss            val accuracy')
print(validationLoss, '  ', validationAccuracy)


# predictions output
predictions = model.predict([xTest])
arr = []
for i in range(20):
    arr.append(np.argmax(predictions[i]))

print('Predictions')
print(arr[0:4])
print(arr[4:8])
print(arr[8:12])
print(arr[12:16])
print(arr[16:20])


# plotting a few images to see what we are working with

fig=plt.figure(figsize=(8, 8))

columns = 4
rows = 5
for i in range(0, columns*rows ):
    img = dat[i]
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img, cmap=plt.cm.binary)
plt.show()



