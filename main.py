import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ##This to prevent info messages from tensorflow
import tensorflow as tf
import argparse
from create_tfrecords import TFrecords_writer
from read_tfrecords import Tfrecord_reader
from train_model import Trainer

parser = argparse.ArgumentParser(description='Creating tfrecords.')
parser.add_argument('--train_loc',type=str, required=True, help='Location of train set images')
parser.add_argument('--test_loc',type=str, required=True, help='Location of test set images')
parser.add_argument('--dest_loc',type=str, required=True, help='Location where tfrecords need to be stored')
parser.add_argument('--compression',type=str,default=None ,help='compression format')
parser.add_argument('--shards',type=bool,default=False ,help='Do you want to shard tfrecords')
parser.add_argument('--train_records_path',type=str,default=None ,help='train records path')
parser.add_argument('--test_records_path',type=str,default=None ,help='test records path')

args = parser.parse_args()

## Limiting the memory used on the device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Creating tfrecords
# tf_writer = TFrecords_writer(args.train_loc,args.dest_loc,args.test_loc,args.compression,args.shards)
# print('Completed creating tfrecords for train and test data')

print('Reading Data......')
if args.train_records_path is None:
    train_path = args.dest_loc+'/train_records'
    test_path = args.dest_loc + '/test_records'

else:
    train_path = self.traon_records_path
    test_path = self.traon_records_path

##Reading TFrecords
train_reader = Tfrecord_reader(tfrecord_path=train_path,shards=args.shards)
train_data = train_reader.load_data()
test_reader = Tfrecord_reader(tfrecord_path=test_path,shards=args.shards)
test_data = test_reader.load_data()
print('Complted Reading Data.')

#Training the model
model_trainer = Trainer(train_data,test_data)
model_trainer.train(epochs=8)
model_trainer.test_model()