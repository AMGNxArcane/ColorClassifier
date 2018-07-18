import tensorflow as tf

from tensorflow import keras
import numpy as np
import json

print(tf.__version__)

labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

data = None
with open('colorData.json') as file:
    data = json.load(file)

colors = []
labels = []
for entry in data['entries']:
    r = entry['r']/255
    g = entry['g']/255
    b = entry['b']/255
    label = entry['label']
    colors.append([r,g,b])
    labels.append(labelList.index(label))

xs = tf.constant(colors)

ys = tf.one_hot(tf.constant(labels),9)
print(xs.shape)
print(ys.shape)



model = keras.Sequential([
    keras.layers.Dense(10,input_shape=[3],activation=tf.nn.relu),
    keras.layers.Dense(9,activation=tf.nn.softmax)
    ])
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(xs,ys,epochs=20,steps_per_epoch=2000)#

model.save('model.h5')
