import numpy as np
import os
from PIL import Image
from mnist import input_data


mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

item = np.reshape(images[0], [28, 28])
item = item * 255.0
image = Image.fromarray(item).convert('L')

img_name = 'train_%s.png' % 28
train_dir = 'y_data/train/7'
# image.save(os.path.join(train_dir, img_name))

filename = 't_data/train/train_28.png'

image = Image.open(filename)
array = np.array(image)

print 222