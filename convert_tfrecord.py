from datetime import datetime
import os
import random
import sys
import threading


import numpy as np
import tensorflow as tf
import configs.config as cfg


num_threads = 4
num_shards = 4

def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename

def _process_image(filename, coder):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    image = image / 255.0

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _convert_to_example(filename, image_buffer, height, width, label):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/labels': _int64_feature(label),
      # 'image/filename': _bytes_feature(filename),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example

def _process_image_files_batch(coder, thread_index, ranges, filenames, labels, num_shards, split_name):
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = 'food_%s_%.5d-of-%.5d.tfrecord' % (split_name, shard, num_shards)
    output_file = os.path.join(cfg.FLAGS.TFrecord_dir +'/'+cfg.FLAGS.surfix, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer,
                                    height, width, label)
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
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(filenames, labels, split_name):
    print('_process_image_files==>')
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, filenames, labels, num_shards, split_name)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(image_dir, train_info, class_to_labels):
    files = []
    labels = []

    with open(train_info, 'r') as txt:
        datas = [l.strip() for l in txt.readlines()]
        for f in datas:
            dir_name, id = f.split('/')
            l = class_to_labels[dir_name]
            files.append(image_dir+f+'.jpg' )
            labels.append(l)

        shuffled_index = list(range(len(files)))

        random.seed(12345)

        random.shuffle(shuffled_index)

        shuffled__files = [files[i] for i in shuffled_index]

        shuffled_labels = [labels[i] for i in shuffled_index]

        return shuffled__files, shuffled_labels

def _process_dataset(dataset_dir, split_name):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  meta_dir = dataset_dir + 'meta/'
  image_dir = dataset_dir + 'images/'
  fp = open(meta_dir + 'classes.txt', 'r')

  classes = [l.strip() for l in fp.readlines()]
  class_to_labels = dict(zip(classes, range(len(classes))))
  # labels_to_class = dict(zip(range(len(classes)), classes))

  filenames, labels = _find_image_files(image_dir, meta_dir+'%s.txt'%split_name, class_to_labels)
  _process_image_files(filenames, labels, split_name)

def main(unused_argv):
    # assert not FLAGS.train_shards % FLAGS.num_threads, (
    #     'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    print('Saving results to %s' % cfg.FLAGS.TFrecord_dir +'/'+cfg.FLAGS.surfix)

    # Run it!
    surfix = cfg.FLAGS.surfix
    _process_dataset(cfg.FLAGS.dataset_dir, surfix)


if __name__ == '__main__':
  tf.app.run()
