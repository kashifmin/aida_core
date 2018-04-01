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


RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip

run pip3 install scandir==1.6  h5py==2.7.1 scikit-image dlib face_recognition
