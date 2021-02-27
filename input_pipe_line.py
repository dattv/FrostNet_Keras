"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def imagenet_data_loader(dataset_dir='./',
                         mode=tf.estimator.ModeKeys.TRAIN,
                         preprocess_fn=None,
                         batch_size=8):
    """

    :param dataset_dir:
    :param mode:
    :param preprocess_fn:
    :param batch_size:
    :return:
    """

    def parser(feature):
        features = tf.io.parse_single_example(
            feature,
            features={
                'image/encoded': tf.compat.v1.FixedLenFeature(
                    (), tf.string, default_value=''),
                'image/format': tf.compat.v1.FixedLenFeature(
                    (), tf.string, default_value='jpeg'),
                'image/class/label': tf.compat.v1.FixedLenFeature(
                    [], dtype=tf.int64, default_value=-1),
                'image/class/text': tf.compat.v1.FixedLenFeature(
                    [], dtype=tf.string, default_value=''),
                'image/object/bbox/xmin': tf.compat.v1.VarLenFeature(
                    dtype=tf.float32),
                'image/object/bbox/ymin': tf.compat.v1.VarLenFeature(
                    dtype=tf.float32),
                'image/object/bbox/xmax': tf.compat.v1.VarLenFeature(
                    dtype=tf.float32),
                'image/object/bbox/ymax': tf.compat.v1.VarLenFeature(
                    dtype=tf.float32),
                'image/object/class/label': tf.compat.v1.VarLenFeature(
                    dtype=tf.int64),
                'image/height':
                    tf.compat.v1.FixedLenFeature((), tf.int64, 1),
                'image/width':
                    tf.compat.v1.FixedLenFeature((), tf.int64, 1),
            }
        )

        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.reshape(image, tf.stack([features['image/height'], features['image/width'], 3]))
        image_name = features['image/filename']
        image_height = features['image/height']
        image_width = features['image/width']

        tensor_dict = {
            'image': image,
            'height': image_height,
            'width': image_width,
            'filename': image_name,
        }

        return tensor_dict

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    autotune = tf.data.experimental.AUTOTUNE
    tfrecord = tf.io.gfile.glob(dataset_dir + '/{}-*'.format(mode.lower()))
    dataset = tf.data.TFRecordDataset(tfrecord)

    dataset = dataset.map(parser, num_parallel_calls=autotune)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    if preprocess_fn is not None:
        dataset = dataset.map(preprocess_fn, num_parallel_calls=autotune)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.shuffle(8 * batch_size)
        dataset = dataset.prefetch(buffer_size=autotune)
        dataset = dataset.repeat()
        # dataset = dataset.cache()

    elif mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)

    else:
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)
    return dataset
