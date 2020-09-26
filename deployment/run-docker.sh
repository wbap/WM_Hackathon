IMG_NAME=wbai/wm:latest

SRC_DIR=~/Dev/wm_hack
TRG_DIR=/home/wm

GPU=False

if $GPU
then
	docker run -dit --rm --name=wm --gpus all --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME
else
	docker run -dit --rm --name=wm --mount type=bind,source=$SRC_DIR,target=$TRG_DIR $IMG_NAME
fi
