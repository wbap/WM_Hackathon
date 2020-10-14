#!/bin/sh

# This requires you to be in the /deployment folder when running
# That is because it sets the context for the Dockerfile

export NAME=${1:-wbai_wm_hackathon}

docker build -t $NAME:latest .
