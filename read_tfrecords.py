import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Tfrecord_reader():
    def __init__(self,tfrecord_path, shards = False, compression = None):
        self.tfrecord_path = tfrecord_path
        self.shards = shards
        self.compression = compression

    def parser_func(self,in_example_protos):
        features_dict = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'class name': tf.io.FixedLenFeature([], tf.string)
        }

        features = tf.io.parse_single_example(in_example_protos, features_dict)

        images = tf.image.resize(tf.io.decode_jpeg(features['image']), (100, 100))  #You can resize all the images to same size here
        label = features['label']
        class_name = features['class name']

        return images, label, class_name

    def load_data(self):
        if self.compression is None:
            comp = 'tfrecords'
        else:
            comp = self.compression
        files_list = tf.data.Dataset.list_files('{}\*.{}'.format(self.tfrecord_path,comp))
        data = tf.data.TFRecordDataset(files_list, compression_type=self.compression)
        dataset = data.map(self.parser_func)
        for i in dataset.take(1):
            plt.imshow(i[0] / 255.0)
            plt.show()
            print(i[2])

        dataset = dataset.shuffle(buffer_size=15000, seed=13)
        dataset = dataset.batch(32, drop_remainder=True)
        return dataset