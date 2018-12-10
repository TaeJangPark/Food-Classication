import glob, os
import tensorflow as tf

from configs import config as cfg
from tensorflow.python.ops import control_flow_ops

_FILE_PATTERN = 'food_%s_*.tfrecord'
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'An labels of color images for classification.',
}

def _flip_image(image):
    return tf.reverse(image, axis=[1])

def _smallest_size_at_least(h, w, np_smallest_side):
    smallest_side = tf.convert_to_tensor(np_smallest_side, dtype=tf.int32)
    h = tf.to_float(h)
    w = tf.to_float(w)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(h, w),
                    lambda : smallest_side / w,
                    lambda : smallest_side / h)
    new_h = tf.to_int32(h * scale)
    new_w = tf.to_int32(w * scale)
    return new_h, new_w

def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def _distorted_bounding_box_crop(image, bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100,
                                 scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def _preprocess_for_training(image, bbox=None):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
    tf.summary.image('image_with_bounding_boxes', image_with_box)

    distorted_image, distorted_bbox = _distorted_bounding_box_crop(image, bbox)
    image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                            distorted_bbox)
    tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)

    distorted_image.set_shape([None, None, 3])
    distorted_image = _apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size], method=method),
        num_cases=1)

    tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 4 ways to do it.
    distorted_image = _apply_with_random_selector(
        distorted_image,
        lambda x, ordering: _distort_color(x, ordering, fast_mode=True),
        num_cases=4)

    tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)

    # ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    # """ step 1. random flipping """
    # thresh = tf.to_float(tf.random_uniform([1]))[0]
    # val = tf.constant(0.5, dtype=tf.float32)
    # image = tf.cond(tf.greater_equal(thresh, val),
    #                 lambda : _flip_image(image),
    #                 lambda : image)
    #
    # """ step 2. min size resize """
    # image = tf.expand_dims(image, 0)
    # image = tf.image.resize_bilinear(image, [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size], align_corners=False)
    # image = tf.squeeze(image, axis=[0])
    #
    # """ step 3. random color distortion """
    # val = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]
    # image = distort_color(image, val)

    # """ step 4. convert rgb to bgr """
    # image = tf.reverse(image, axis=[-1])

    return distorted_image

def _preprocess_for_test(image, central_fraction=0.875):
    with tf.name_scope('test_image'):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image,
                                         [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

def _batcher(filename, batch_size, is_training, num_epoches=30, min_after_dequeue=16):
    if not isinstance(filename, list):
        filename = [filename]

    print("data_batch for food classification ==> !!!!")
    if is_training:
        num_epoches = cfg.FLAGS.num_epochs
    else:
        num_epoches = 1
    filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epoches)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example, features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/labels': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string)
        })

    ih = tf.cast(features['image/height'], tf.int32)
    iw = tf.cast(features['image/width'], tf. int32)
    labels = tf.cast(features['image/labels'], tf.int32)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    """ preprocessing which is random flipping, min size resizeing and zero mean """
    if is_training is True:
        print("Start ==> preprocessing for training")
        image = _preprocess_for_training(image)
    else:
        print("Start ==> preprocessing for test")
        image = _preprocess_for_test(image)

    """ generate batch data """
    min_after_dequeue = 5 * batch_size
    capacity = min_after_dequeue + 10 * batch_size
    example_batch, labels_batch = tf.train.shuffle_batch([image, labels],
                                                         batch_size=batch_size,
                                                         capacity=capacity,
                                                         min_after_dequeue=min_after_dequeue,
                                                         num_threads=cfg.FLAGS.num_preprocessing_threads)

    tf.summary.image('shuffled_images', example_batch)

    return(example_batch, labels_batch)



def get_dataset(split_name,
                dataset_dir,
                is_training,
                im_batch=4,
                file_pattern=None):
    if file_pattern is None:
        file_pattern = _FILE_PATTERN % (split_name)

    path = os.path.join(dataset_dir, file_pattern)
    filenames = glob.glob(path)
    return _batcher(filenames, im_batch, is_training)


