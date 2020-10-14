#!/bin/sh

export NAME=${1:-wbai_wm_hackathon}

docker build -t $NAME:latest .
