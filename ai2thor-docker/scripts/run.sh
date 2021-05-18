#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR/../

export ROBOTHOR_BASE_DIR=`pwd`

# Inference on train split
X11_PARAMS=""
if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
  echo "Using local X11 server"
  X11_PARAMS="-e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
  xhost +local:root
fi;

docker run --privileged $X11_PARAMS -d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 --name=roy_ai2thor-cu10 -v /home/container-data/roy_ai2thor/:/home/store -p 39907:22 -v /home/roy/:/home/ -it ai2thor-docker-cu10.1:latest

if [[ -e /tmp/.X11-unix && ! -z ${DISPLAY+x} ]]; then
    xhost -local:root
fi;

