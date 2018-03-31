from tensorflow/tensorflow:latest-py3


copy ./requirements.txt /requirements.txt
run cd / && pip3 install -r requirements.txt

run mkdir /usr/src/app
workdir /usr/src/app/

run apt update
run apt install python3-tk --yes

run apt install git --yes

run git clone https://github.com/tensorflow/models.git

workdir /usr/src/app/models/research

run apt-get install protobuf-compiler --yes

run protoc object_detection/protos/*.proto --python_out=.



cmd cd /usr/src/app/models/research && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

workdir /usr/src/aida

run pip3 install opencv-python

run apt install --yes libsm6 libxext6


workdir /usr/src/app/

run apt install wget --yes

run wget "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
run tar -xzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
run rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
run mv faster_rcnn_inception_v2_coco_2018_01_28 coco_inception


workdir /usr/src/aida

run pip install requests