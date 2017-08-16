# -*- coding:utf-8 -*-
# import tensorflow as tf
# from openHPC_web_project.webconsole.views.ai_views.frameworks.tensorflow.tensorflow_proto_temps import \
#     lenet

import numpy as np
from PIL import Image
from datetime import datetime
import os
import random
import sys
import threading
import time

import tensorflow as tf
import input_data
import lenet

mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)

# step 1 : get data set files
# step 2 : pre-treat images to match the topology's requirement
# step 3 : generate binary files for those data set files for the usage of tensorflow's training progress
# step 4 : generate label files
# step 5 : run the tensorflow training progress and generate formatted output log file for the model
# step 6 : parsing output file
# step 7 : Test the model after training finished

# request args
train_args = {}
dataset_args = {}

training_args_default = {
    "train_epochs": 30,

    "train_batch_size": 128,
    "test_batch_size": 128,
    "validation_batch_size": 128,

    "validation_interval": 1,
    "snapshot_interval": 1,

    "regularization_type": 'L2',
    "regularization_rate": 0.0001,

    "base_lr": 0.001,
    "lr_decay_rate": 0.99,

    "max_steps": 10000
}

dataset_args_default = {
    "image_size": 28,
    "topology": "Lenet",

    "data_dir": "data",
    "working_dir": "output",

    "train_shards": 1,
    "test_shards": 1,
    "num_threads": 1,
}


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label):
    """Build an Example proto for an example.
    Args:
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    Returns:
    Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_buffer)
    }))
    return example


def _process_image_files_batch(thread_index, ranges, name, filenames,
                               labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
    Args:
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
    analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_dir = os.path.join(train_args["working_dir"], "data")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = int(labels[i])

            image = Image.open(filename)
            array = np.array(image)
            image_buffer = array.tostring()
            example = _convert_to_example(image_buffer, label)
            writer.write(example.SerializeToString())

            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def generate_binary_for_dataset(name, filenames, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), train_args["num_threads"] + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (train_args["num_threads"], ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(list_file, data_dir):
    print('Determining list of input files and labels from %s.' % list_file)
    files_labels = [l.strip().split(' ') for l in tf.gfile.FastGFile(
        list_file, 'r').readlines()]

    labels = []
    filenames = []

    # Construct the list of JPEG files and labels.
    for path, label in files_labels:
        jpeg_file_path = '%s/%s' % (data_dir, path)
        if os.path.exists(jpeg_file_path):
            filenames.append(jpeg_file_path)
            labels.append(label)

    unique_labels = set(labels)
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames, labels


def _process_dataset(name, filename, directory, num_shards):
    """Process a complete data set and save it as a TFRecord.
    Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
    """
    filenames, labels = _find_image_files(filename, directory)
    generate_binary_for_dataset(name, filenames, labels, num_shards)


def pre_treat_images():
    _process_dataset('test', os.path.join(train_args["data_dir"], "test", "test.txt"), train_args["data_dir"] + "/test",
                     train_args["test_shards"])
    _process_dataset('train', os.path.join(train_args["data_dir"], "train", "train.txt"), train_args["data_dir"] + "/train",
                                             train_args["train_shards"])


def data_files(data_dir, subset):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if subset not in ['train', 'validation', 'test']:
        print('Invalid subset!')
        exit(-1)

    tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
    data_files = tf.gfile.Glob(tf_record_pattern)
    print(data_files)
    if not data_files:
        print('No files found for data dir %s at %s' % (subset, data_dir))

        exit(-1)
    return data_files


def get_dataset_files():
    return ""


def read_data_sets(dataset_binary_file):
    files = tf.train.match_filenames_once(dataset_binary_file)
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    # read the file bt tensorflow
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # parse the example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       })

    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images_tmp = tf.multiply(retyped_images, 1.0 / 255.0)
    images = tf.reshape(images_tmp, [784])

    labels = tf.cast(features['label'], tf.int32)
    label = tf.stack(tf.one_hot(labels, 10))

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * train_args['train_batch_size']

    image_batch, label_batch = tf.train.shuffle_batch([images, label],
                                                      batch_size=train_args['train_batch_size'],
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    image_batch = tf.reshape(image_batch,
                             [train_args['train_batch_size'], train_args['image_size'], train_args['image_size'], 1])

    return image_batch, label_batch


class DataSet(object):
  def __init__(self, images, labels, dtype=tf.float32):
    self._images = images
    self._labels = labels

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels


def get_dataset_batch():
    class DataSets(object):
        pass

    data_sets = DataSets()

    train_binary_file = data_files(os.path.join(train_args["working_dir"], "data"), 'train')
    # val_binary_file = data_files(os.path.join(train_args["working_dir"], "data"), 'validation')
    test_binary_file = data_files(os.path.join(train_args["working_dir"], "data"), 'test')

    train_image_batch, train_label_batch = read_data_sets(train_binary_file)
    # val_image_batch, val_label_batch = read_data_sets(val_binary_file)
    test_image_batch, test_label_batch = read_data_sets(test_binary_file)

    data_sets.train = DataSet(train_image_batch, train_label_batch)
    # data_sets.validation = DataSet(val_image_batch, val_label_batch)
    data_sets.test = DataSet(test_image_batch, test_label_batch)

    return data_sets


def train(dataset, total_dataset_columns):
    x = tf.placeholder(tf.float32, [None, train_args['image_size'], train_args['image_size'], 1])
    y_ = tf.placeholder(tf.float32, [None, 10])

    logits, _ = lenet.lenet(x)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.train.exponential_decay(
        train_args['base_lr'],
        global_step,
        total_dataset_columns / train_args['train_batch_size'], train_args['lr_decay_rate'],
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    steps_per_train_epoch = int(total_dataset_columns / train_args['train_batch_size'])

    num_steps = train_args['max_steps'] if train_args['train_epochs'] < 1 else train_args[
                                                                                   'train_epochs'] * steps_per_train_epoch

    print('Requested number of steps [%d]' % num_steps)

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(os.path.join(train_args['working_dir'], "summary_logs"), sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(num_steps):
            batch_xs, batch_ys = sess.run([dataset.train.images, dataset.train.labels])
            start_time = time.time()
            _, loss_value, global_step_val = sess.run([train_step, loss, global_step], feed_dict={x: batch_xs, y_: batch_ys})

            duration = time.time() - start_time

            if step % 10 == 0:
                num_examples_per_step = train_args['train_batch_size']
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            # validation interval and set result into redis
            if (step + 1) % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_xs, y_: batch_ys})

                vbatch_xs, vbatch_ys = sess.run([dataset.test.images, dataset.test.labels])
                validation_accuracy = accuracy.eval(feed_dict={
                    x: vbatch_xs, y_: vbatch_ys})
                print("step %d, validation accuracy %g" % (step + 1, validation_accuracy))


                print("step %d, training accuracy %g" % (step + 1, train_accuracy))

                learning_rate_1 = sess.run(learning_rate)
                print("step %d, learning_rate %g" % (step + 1, learning_rate_1))

            # snapshot interval and set result into redis
            if (step + 1) % 1000 == 0:
                saver.save(sess, os.path.join(train_args['working_dir'], "checkpoint"), global_step=global_step)

        # test model and set result into redis

        coord.request_stop()
        coord.join(threads)


def parse_output():
    pass


def test_model():
    pass


def main(argv=None):
    # request args check and make up, train_args would be the final and cover all request args
    train_args.update(training_args_default)
    dataset_args.update(dataset_args_default)
    train_args.update(dataset_args)

    # step 1
    # dataset_files = get_dataset_files()

    # step 2
    # pre_treat_images()

    # step 3

    # step 4
    dataset = get_dataset_batch()

    # step 5
    train(dataset, 10000)


if __name__ == '__main__':
    main()
