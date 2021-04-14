import numpy as np
import tensorflow as tf
import os

class TFrecords_writer():
    def __init__(self,train_loc,destination,test_loc=None,compression=None,sharding = False):
        self.train_images_loc = train_loc
        self.test_images_loc = test_loc
        self.destination_loc = destination
        self.sharding = False
        self.compression = compression

        self.create_tfrecords_train()
        if self.test_images_loc is not None:
            self.create_tfrecords_test()


    def  _bytes_feature(self,value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def img_to_record(self,img_string, label, class_name):
        feature_dict = {
            'image': self._bytes_feature(img_string),
            'label': self._int64_feature(label),
            'class name': self._bytes_feature(class_name)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tfrecords_train(self):
        print('The number of classes are: ' + str(len(os.listdir(self.train_images_loc))))
        nums = {}
        for i in os.listdir(self.train_images_loc):
            nums[i] = len(os.listdir(self.train_images_loc + '/' + i))
        print('Number of images in each class.')
        for i in nums.keys():
            print('{} : {}'.format(i, nums[i]))

        if self.sharding:
            if self.compression is not None:
                self.compression = tf.io.TFRecordOptions(compression_type=self.compression)
            tfrecords_shard_path = self.destination_loc + '/{}.{}'.format('train','tfrecords')
            lbl = 0
            for cls in os.listdir(self.train_images_loc):
                tfrecords_shard_path = self.destination_loc + '/{}.{}'.format(cls, 'tfrecords')
                with tf.io.TFRecordWriter(tfrecords_shard_path,options=self.compression) as writer:
                    for img in os.listdir(self.train_images_loc+'/'+cls):
                        img_loc = self.train_images_loc + '/' + cls + '/' + img
                        img_string = open(img_loc, 'rb').read()
                        tf_ex = self.img_to_record(img_string, lbl, str.encode(cls))
                        writer.write(tf_ex.SerializeToString())

                    lbl+=1

        else:
            if self.compression is not None:
                self.compression = tf.io.TFRecordOptions(compression_type=self.compression)
            tfrecords_shard_path = self.destination_loc + '/{}.{}'.format('train','tfrecords')
            lbl = 0
            with tf.io.TFRecordWriter(tfrecords_shard_path,options=self.compression) as writer:
                for cls in os.listdir(self.train_images_loc):
                    for img in os.listdir(self.train_images_loc+'/'+cls):
                        img_loc = self.train_images_loc + '/' + cls + '/' + img
                        img_string = open(img_loc, 'rb').read()
                        tf_ex = self.img_to_record(img_string, lbl, str.encode(cls))
                        writer.write(tf_ex.SerializeToString())

                    lbl+=1

    def create_tfrecords_test(self):
        nums = {}
        for i in os.listdir(self.test_images_loc):
            nums[i] = len(os.listdir(self.test_images_loc + '/' + i))
        print('Number of test images in each class.')
        for i in nums.keys():
            print('{} : {}'.format(i, nums[i]))

        if self.sharding:
            if self.compression is not None:
                self.compression = tf.io.TFRecordOptions(compression_type=self.compression)
            tfrecords_shard_path = self.destination_loc + '/{}.{}'.format('test', 'tfrecords')
            lbl = 0
            for cls in os.listdir(self.test_images_loc):
                tfrecords_shard_path = self.destination_loc + '/{}.{}'.format(cls, 'tfrecords')
                with tf.io.TFRecordWriter(tfrecords_shard_path, options=self.compression) as writer:
                    for img in os.listdir(self.test_images_loc + '/' + cls):
                        img_loc = self.test_images_loc + '/' + cls + '/' + img
                        img_string = open(img_loc, 'rb').read()
                        tf_ex = self.img_to_record(img_string, lbl, str.encode(cls))
                        writer.write(tf_ex.SerializeToString())

                    lbl += 1

        else:
            if self.compression is not None:
                self.compression = tf.io.TFRecordOptions(compression_type=self.compression)
            tfrecords_shard_path = self.destination_loc + '/{}.{}'.format('test', 'tfrecords')
            lbl = 0
            with tf.io.TFRecordWriter(tfrecords_shard_path, options=self.compression) as writer:
                for cls in os.listdir(self.test_images_loc):
                    for img in os.listdir(self.test_images_loc + '/' + cls):
                        img_loc = self.test_images_loc + '/' + cls + '/' + img
                        img_string = open(img_loc, 'rb').read()
                        tf_ex = self.img_to_record(img_string, lbl, str.encode(cls))
                        writer.write(tf_ex.SerializeToString())

                    lbl += 1



