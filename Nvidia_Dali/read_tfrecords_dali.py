from subprocess import call

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np


def create_idx_file(tfrecord_path,dest_path):
    call(['tfrecord2idx', tfrecord_path, dest_path])


def dali_tfrecords(tfrecord_path,batch_size=32,num_threads=4):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrecord_path,
            index_path=tfrecord_idx,
            features={
                "image": tfrec.FixedLenFeature((), tfrec.string, ""),
                "label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
                "class name": tfrec.FixedLenFeature([], tfrec.string, "")})
        image = inputs["image"]
        image = fn.decoders.image(image, device="mixed", output_type=types.RGB)
        #resized = fn.resize(images, device="gpu", resize_shorter=256.)
        label = inputs['label']
        pipe.set_outputs(image, inputs['label'])