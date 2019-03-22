import pygame
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras

model = keras.models.load_model("model.h5")

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


pygame.init()
pygame.display.set_caption("Color Classifier")
screen = pygame.display.set_mode([400,400])
r = random.randint(0,255)
g = random.randint(0,255)
b = random.randint(0,255)

font = pygame.font.SysFont("monospace",32)


tick = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    screen.fill((r,g,b))
    xs = np.array([r/255.0,g/255.0,b/255.0]).reshape(-1,3)
    result = keras.backend.argmax(model.predict(xs))
    #print(r,g,b)
    with tf.Session() as sess:
        index = sess.run(result)
        #print(index)
        print(labelList[index[0]])
        label = font.render(labelList[index[0]],1,(0,0,0))
        screen.blit(label,(0,0))

    pygame.display.update()
    tick.tick(1)
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
