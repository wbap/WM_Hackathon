FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

USER root

# OS Dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        libgl1-mesa-dev \
        git \
        tcl-dev \
        tk-dev \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure that cuDNN is detected
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so

# Set the working directory to
ENV HOME /root
WORKDIR $HOME

# Create volumes (that we'll map to when we run the container)
RUN mkdir -p $HOME/wm_hackathon
VOLUME $HOME/wm_hackathon

# Copy Conda environment config
COPY environment.yml ./environment.yml

# Install Miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p $HOME/miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH $HOME/miniconda/bin:$PATH

# Create a Conda environment
RUN conda env create -f environment.yml -n wm_env
ENV PATH $HOME/miniconda/envs/wm_env/bin:$PATH

# Clean-up
RUN conda clean --all -y

RUN conda init bash
RUN . $HOME/.bashrc
RUN conda activate wm_env
RUN git clone https://github.com/Cerenaut/cerenaut-pt-core.git
RUN python cerenaut-pt-core/setup.py develop

# TensorBoard
EXPOSE 6006

# Setup .bashrc
RUN echo "export LANG=C.UTF-8" >> $HOME/.bashrc
RUN echo "export LC_ALL=C.UTF-8" >> $HOME/.bashrc
RUN echo "export PATH=/usr/local/cuda-9.0/bin:$PATH" >> $HOME/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" >> $HOME/.bashrc

# set the default path to the mounted folder (commands passed to the container are run from there)
WORKDIR $HOME/wm_hackathon

CMD ["bash"]
