from tensorflow/tensorflow:latest-py3


copy ./requirements.txt /requirements.txt
run cd / && pip3 install -r requirements.txt

run mkdir /usr/src/app
workdir /usr/src/app/

run apt update
run apt install python3-tk --yes

run apt install git --yes


workdir /usr/src/app/models/research

run apt-get install protobuf-compiler --yes

run protoc object_detection/protos/*.proto --python_out=.

cmd export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

workdir /usr/src/aida

run pip3 install opencv-python

run apt install --yes libsm6 libxext6

run apt install wget --yes

run pip3 install requests