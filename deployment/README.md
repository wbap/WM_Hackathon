Prereqs:
- Create group incubator (gid=4910) on remote machine. We'll use this to try and link permissions between host and container users.
- Ensure NVIDIA driver is installed (NOTE: THIS DOES NOT MEAN CUDA)

## Required Dependencies
Get the following package binaries for the remote machine's distribution. Hierarchy represents the dependency.

- docker-ce
-- docker-ce-cli
---- containerd.io

- nvidia-docker
-- nvidia-container-runtime
---- nvidia-container-toolkit
------ libnvidia-container-tools
-------- libnvidia-container1

**Note:** For this guide, we'll use `ubuntu18.04/amd64` as an example, given that it's the most common distribution.

## Setting Up Docker
1. Install `containerd.io`, a Docker dependency, using `sudo apt install ./containerd.io_1.2.13-1_amd64.deb`
2. Install `docker-ce-cli` using `sudo apt install ./docker-ce-cli_19.03.8~3-0~ubuntu-bionic_amd64.deb`
3. Install `docker-ce` using `sudo apt install ./docker-ce-cli_19.03.8~3-0~ubuntu-bionic_amd64.deb`
4. Verify that Docker is installed & running using `docker ps`

## Setting Up NVIDIA Runtime
1. Install libnvidia-container1 using `sudo apt install ./libnvidia-container1_1.0.7-1_amd64.deb`
2. Install libnvidia-container-tools using `sudo apt install ./libnvidia-container-tools_1.0.7-1_amd64.deb`
3. Install nvidia-container-toolkit using `sudo apt install ./nvidia-container-toolkit_1.0.5-1_amd64.deb`
4. Install nvidia-container-runtime using `sudo apt install ./nvidia-container-runtime_3.1.4-1_amd64.deb`
5. Install nvidia-docker using `sudo apt install ./nvidia-docker2_2.2.2-1_all.deb`

## Building Docker Image
- Build image locally using `docker build -t IMAGE_NAME:latest .` (this step may take a while, especially first time)
- Save image to archive (TAR) using `docker save IMAGE_NAME:latest > image.tar`
- Copy the image archive to the remote machine
- Load the Docker image on remote machine using `docker load < image.tar`
- Start new containers using `docker run IMAGE_NAME:latest`
