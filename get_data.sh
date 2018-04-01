#!/bin/sh
mkdir weights
wget "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
tar -xzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
mv faster_rcnn_inception_v2_coco_2018_01_28 weights

# get tensorflow models
git clone https://github.com/tensorflow/models.git tfmodels
echo "Downloaded all requirements successfully"
