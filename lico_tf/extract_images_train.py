import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from mnist import input_data
from PIL import Image

train_dir = 'data/train/'
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
train_file = open(os.path.join(train_dir, 'train.txt'), 'w')

mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
num_examples = mnist.train.num_examples

for index in range(num_examples):

    item = np.reshape(images[index], [28, 28])
    item = item * 255
    image = Image.fromarray(item).convert('L')

    label = str(np.argmax(labels[index]))

    img_name = 'train_%s.png' % index
    dirname = train_dir + label
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    image.save(os.path.join(dirname, img_name))
    # plt.axis('off')
    # plt.imshow(item, cmap='binary')
    # plt.savefig(os.path.join(dirname, img_name))

    label = '%s/%s %s\n' % (label, img_name, label)
    train_file.write(label)

    if index % 100 == 0:
        print("This is at step: %s" % index)

