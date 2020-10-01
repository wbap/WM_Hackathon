#!/bin/sh

# usage: run-docker.sh [path to wm code] [boolean for GPU or not] [command and params to run in container]
# e.g. ../deployment/run-docker.sh ~/Dev/wm_hack False python train_stub_agent.py dm2s-v0 configs/dm2s_env.par configs/simple_agent_model.json
# ../deployment/run-docker.sh ~/Dev/wm_hack False python keyboard_agent.py dm2s-v0 configs/dm2s_env.par

export SRC_DIR=${1:-~/Dev/wm_hack}
export GPU=${2:-False}

shift
shift

TRG_DIR=/root/wm
IMG_NAME=wbai/wm:latest

if $GPU
then
	GPU_STR="--gpus all"
fi

# detached (-d) and remove it when complete (-rm)
# With -rm, we cannot see logs after the container closed
# cmd="docker run -dit --rm --name=wm $GPU_STR --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME $@"

# not detached (i.e. see the output)
cmd="docker run -it --rm --name=wm $GPU_STR --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME $@"
echo $cmd
eval $cmd
