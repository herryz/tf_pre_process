import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from mnist import input_data
from PIL import Image

train_dir = 'data/mnist/'
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
train_num_examples = mnist.train.num_examples

for index in range(train_num_examples):

    item = np.reshape(images[index], [28, 28])
    item = item * 255
    image = Image.fromarray(item).convert('L')

    label = str(np.argmax(labels[index]))

    img_name = 'train_%s.png' % index
    dirname = train_dir + label
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    image.save(os.path.join(dirname, img_name))

    if index % 100 == 0:
        print("This is at step: %s" % index)


images = mnist.test.images
labels = mnist.test.labels
num_examples = mnist.test.num_examples

for index in range(num_examples):

    item = np.reshape(images[index], [28, 28])
    item = item * 255
    image = Image.fromarray(item).convert('L')

    label = str(np.argmax(labels[index]))

    img_name = 'train_%s.png' % str(index + train_num_examples)
    dirname = train_dir + label
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    image.save(os.path.join(dirname, img_name))

    if index % 100 == 0:
        print("This is at step: %s" % str(index + train_num_examples))