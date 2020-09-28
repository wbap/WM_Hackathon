#!/bin/sh

# usage: run-docker.sh [path to wm code] [boolean for GPU or not]

export SRC_DIR=${1:-~/Dev/wm_hack}
export GPU=${2:-False}

TRG_DIR=/root/wm
IMG_NAME=wbai/wm:latest

if $GPU
then
	GPU_STR="--gpus all"
fi

docker run -dit --rm --name=wm $GPU_STR --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME
