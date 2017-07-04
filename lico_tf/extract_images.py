import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from mnist import input_data
from PIL import Image

test_dir = 'y_data/test/'
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
test_file = open(os.path.join(test_dir, 'test.txt'), 'w')

mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)
images = mnist.test.images
labels = mnist.test.labels
num_examples = mnist.test.num_examples

for index in range(2):

    item = np.reshape(images[index], [28, 28])
    item = item * 255
    image = Image.fromarray(item).convert('L')

    label = str(np.argmax(labels[index]))

    img_name = 'test_%s.png' % index
    dirname = test_dir + label

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    image.save(os.path.join(dirname, img_name))
    # plt.axis('off')
    # plt.imshow(item, cmap='binary')
    # plt.savefig(os.path.join(dirname, img_name))

    label = '%s/%s %s\n' % (label, img_name, label)
    test_file.write(label)

    if index % 100 == 0:
        print("This is at step: %s" % index)
