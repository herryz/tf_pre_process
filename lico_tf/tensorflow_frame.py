# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import os
import random
import sys
import threading
import Lenet

tf.app.flags.DEFINE_integer('train_epochs', 20,
                            'Train epoch')
tf.app.flags.DEFINE_integer('train_batch_size', 128,
                            'Train batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 128,
                            'Test batch size')
tf.app.flags.DEFINE_integer('validation_batch_size', 128,
                            'Validation batch size')
tf.app.flags.DEFINE_integer('validation_interval', 1,
                            'Validation interval')
tf.app.flags.DEFINE_integer('snapshot_interval', 2,
                            'Snapshot interval')
tf.app.flags.DEFINE_string('regularization_type', 'L2',
                            'Regularization type')
tf.app.flags.DEFINE_float('regularization_rate', 0.0001,
                            'Regularization rate')
tf.app.flags.DEFINE_string('base_lr', 0.0,
                           'Training directory')
tf.app.flags.DEFINE_float('lr_decay_rate', 0.99,
                            'Batch size')
tf.app.flags.DEFINE_string('train_dir', '/opt/tf_pro/lico_tf/output/train-*',
                           'Training directory')
tf.app.flags.DEFINE_integer('image_size', 28,
                            'Image size')
tf.app.flags.DEFINE_string('topology', 'Lenet',
                            'Topology')
tf.app.flags.DEFINE_string('max_steps', 10000,
                            'Max steps')

tf.app.flags.DEFINE_string('data_dir', 'data/train',
                           'Data directory')
tf.app.flags.DEFINE_string('output_dir', 'output',
                           'Output directory')
tf.app.flags.DEFINE_string('train_list', 'train.txt',
                           'Training list')
tf.app.flags.DEFINE_string('test_list', 'test.txt',
                           'Test list')
tf.app.flags.DEFINE_integer('train_shards', 10,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 2,
                            'Number of shards in test TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')
FLAGS = tf.app.flags.FLAGS


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
        output_file = os.path.join(FLAGS.output_dir, output_filename)
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


def _process_image_files(name, filenames, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
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
    _process_image_files(name, filenames, labels, num_shards)
    unique_labels = set(labels)
    return len(labels), unique_labels


def prepare():
    # test, test_outcomes = _process_dataset('test', '%s/%s' % (FLAGS.data_dir, FLAGS.test_list), FLAGS.data_dir,
    #                                          FLAGS.valid_shards)
    train, train_outcomes = _process_dataset('train', '%s/%s' % (FLAGS.data_dir , FLAGS.train_list), FLAGS.data_dir,
                      FLAGS.train_shards)


def get_data():
    files = tf.train.match_filenames_once(FLAGS.train_dir)
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    labels = tf.cast(features['label'], tf.int32)
    images = tf.reshape(retyped_images, [784])

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * FLAGS.train_batch_size

    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                      batch_size=FLAGS.train_batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    image_batch = tf.reshape(image_batch, [FLAGS.train_batch_size, FLAGS.image_size, FLAGS.image_size, 1])

    return image_batch, label_batch


def train():
    image_batch, label_batch = get_data()

    logits, _ = Lenet.lenet(image_batch)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean

    learning_rate = tf.train.exponential_decay(
        FLAGS.base_lr,
        global_step,
        10000/ FLAGS.train_batch_size, FLAGS.lr_decay_rate,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    steps_per_train_epoch = int(FLAGS.max_steps / FLAGS.train_epochs)
    num_steps = FLAGS.max_steps if FLAGS.train_epochs < 1 else FLAGS.train_epochs * steps_per_train_epoch
    print('Requested number of steps [%d]' % num_steps)

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(num_steps):
            # sess.run(train_step)
            _, loss_value, global_step_val = sess.run([train_step, loss, global_step])

            # validation interval and set result into redis
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # snapshot interval and set result into redis
            if step % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, FLAGS.train_dir, global_step=global_step)

        # test model and set result into redis

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # prepare()
    train()

if __name__ == '__main__':
    main()