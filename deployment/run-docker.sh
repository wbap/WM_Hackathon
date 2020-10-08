#!/bin/sh

# usage: run-docker.sh [path to wm code] [boolean for GPU or not] [command and params to run in container]
# e.g. ../deployment/run-docker.sh ~/Dev/wm_hack ~/Dev/cerenaut-pt-core False python train_stub_agent.py dm2s-v0 configs/dm2s_env.par configs/simple_agent_model.json
# ../deployment/run-docker.sh ~/Dev/wm_hack ~/Dev/cerenaut-pt-core False python keyboard_agent.py dm2s-v0 configs/dm2s_env.par
# ../deployment/run-docker.sh ~/Dev/wm_hack ~/Dev/cerenaut-pt-core False bash run_kb.sh
export WM_SRC_DIR=${1:-~/Dev/wm_hack}
export CORE_SRC_DIR=${2:-~/Dev/cerenaut-pt-core}
export GPU=${3:-False}

shift
shift
shift

WM_TGT_DIR=/root/wm_hackathon
CORE_TGT_DIR=/root/cerenaut-pt-core
IMG_NAME=wbai/wm:latest

if $GPU
then
	GPU_STR="--gpus all"
fi

# detached (-d) and remove it when complete (-rm)
# With -rm, we cannot see logs after the container closed
# cmd="docker run -dit --rm --name=wm $GPU_STR --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME $@"

# not detached (i.e. see the output)
# cmd="docker run --privileged -it --rm --name=wm $GPU_STR --mount type=bind,source=$SRC_DIR,target=$TGT_DIR $IMG_NAME $@"


cmd="docker run --privileged -it --rm --name=wm $GPU_STR -e DISPLAY=$IP:0 -e XAUTHORITY=/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/.Xauthority:/.Xauthority -v $WM_SRC_DIR:$WM_TGT_DIR -v $CORE_SRC_DIR:$CORE_TGT_DIR $IMG_NAME $@"

echo $cmd
eval $cmd
