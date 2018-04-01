from gcr.io/tensorflow/tensorflow:latest-gpu-py3


copy ./requirements.txt /requirements.txt
run cd / && pip3 install -r requirements.txt

run mkdir /usr/src/app
workdir /usr/src/app/

run apt update
run apt install python3-tk --yes

run apt install git --yes

run apt-get install protobuf-compiler --yes

workdir /usr/src/aida

run pip3 install opencv-python

run apt install --yes libsm6 libxext6

run apt install wget --yes

run pip3 install requests