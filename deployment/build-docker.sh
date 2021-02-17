#!/bin/sh

# This requires you to be in the /deployment folder when running
# That is because it sets the context for the Dockerfile

export NAME=${1:-cerenaut/wbaiwmhackathon}

docker build -t $NAME:latest .
