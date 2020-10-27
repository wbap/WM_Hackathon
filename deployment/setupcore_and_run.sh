#!/bin/sh

# this script is used by run_in_docker.sh to first setup cerenaut-pt-core, and then execute the chosen command

# install dependent project of core modules, into the environment
cd ../cerenaut-pt-core
python setup.py develop
cd ../wm_hackathon/

source activate wm_env

echo $@
eval $@