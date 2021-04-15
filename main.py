import tensorflow as tf
import argparse
from create_tfrecords import TFrecords_writer
from read_tfrecords import Tfrecord_reader

parser = argparse.ArgumentParser(description='Creating tfrecords.')
parser.add_argument('--train_loc',type=str, default='D:\Truveta\Data\seg_train\seg_train', help='Location of train set images')
parser.add_argument('--test_loc',type=str, default=None, help='Location of test set images')
parser.add_argument('--dest_loc',type=str, default='D:\Truveta\Shards', help='Location where tfrecords need to be stored')
parser.add_argument('--compression',type=str,default=None ,help='compression format')
parser.add_argument('--shards',type=bool,default=False ,help='Do you want to shard tfrecordws')
parser.add_argument('--train_records_location', type=str, help='Location where your trainset tfrecords are located')
parser.add_argument('--test_records_location', type=str, help='Location where your testset tfrecords are located')

args = parser.parse_args()


# Creating tfrecords
#tf_writer = TFrecords_writer(args.train_loc,args.dest_loc,args.test_loc,args.compression,args.shards)
#print(tf_writer)
#tf_writer.create_tfrecords()

##Reading TFrecords
reader = Tfrecord_reader(tfrecord_path='D:\Truveta\Shards',shards=args.shards)
train_data = reader.load_data()
