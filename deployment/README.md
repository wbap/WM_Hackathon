# WBAI WM Hackathon Dev Environment using Docker

## Requirements
- Docker
    - Follow instructions on [website](https://www.docker.com) to set it up locally (depends on your environment)
- X11 if you want to display a GUI

- Source code
	- WM_Hackathon [repository](https://github.com/wbap/WM_Hackathon)
	- cerenaut-pt-core [repository](https://github.com/Cerenaut/cerenaut-pt-core)

## Background

The WM_Hackathon project contains a basic agent that can solve the match the sample task. It is provided as a basis for improved agents.
This deployment solution allows you to run the software on any machine running docker. The environment is set up for you already. 
You will have a local copy of the code for development with your preferred tools. It will be mounted inside the container for running.


There is a Docker Image called wb_wm_hackathon available on Docker Hub. It contains te environment setup to run the software. 
When you run a command using the scripts in this folder, the Image will be automatically downloaded, cached on your local machine, and used to create a container. Your commands will then be run inside the container.

The container allows you to utilise the GPU if your machine has one.
It will also by default, utilise X11 on your machine to display the GUI. Note that this is not supported for Macs.
If you are using a Mac, you can easily create a conda environment to run the software (more below)


## Getting Started

1. Clone the code
	- WM_Hackathon repository
	- cerenaut-pt-core repository
3. Running the code
    - All you need to do is pass a given command to the script `bash run-docker.sh` to run it in the container.
    - The full usage instructions are:
`run-docker.sh [path to wm code] [boolean for GPU or not] [command and params to run in container]`


Examples:

`../deployment/run-docker.sh ~/Dev/WM_Hackathon ~/Dev/cerenaut-pt-core False python keyboard_agent.py m2s-v0 configs/m2s_env.json`

`../deployment/run-docker.sh ~/Dev/WM_Hackathon ~/Dev/cerenaut-pt-core False python train_stub_agent.py m2s-v0 configs/m2s_env.json configs/stub_model_full.json configs/stub_agent_full.json` 



## Custom Container
You can modify and build your own dev environment.
First modify the Dockerfile to your spec. 
Then use the script `build-docker.sh [image name]` to build a local copy of the image.

You can then modify the `run-docker.sh` script to refer to your custom image name, and run as above.


# Conda environment for display of GUI on a Mac

## Requirements
- Conda
    - Follow instructions on [website](https://www.anaconda.com) to set it up locally
   
## Instructions
1. The file `environment.yaml` defines all of the dependencies that are required.
You can use it to create a conda environment with the command below, see instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).:

    `conda env create -f environment.yml`.

2. Then, when you want to dev and run, change into the environment. This is a platform specific command. It is usually:

    `source activate [env-name]`

3. Finally, you need to setup a dependency, the project `cerenaut-pt-core` by following the `README.md`. In brief, navigate to the folder and then use the following command:

    `python setup.py develop` 