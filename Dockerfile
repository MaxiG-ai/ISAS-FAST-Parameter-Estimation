FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 as base

LABEL Name=cempc Version=0.0.1
ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    texlive \
    texlive-latex-extra \
    wget \
    bzip2 \
    ca-certificates \
    git \
    bash \
    curl \
    gmsh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Optional: install Python 3.12 if you need it outside of conda
#RUN apt-get install -y python3.12 python3-pip

# Install Miniconda (will be used instead of system Python)
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -u -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Copy environment file and create conda environment
#COPY environment.yml .
#RUN conda init bash && \
#    source $CONDA_DIR/etc/profile.d/conda.sh && \
#    conda update -n base -c defaults conda && \
#    conda env create -f environment.yml -p /opt/conda/envs/jax-fem 
#    && \
#    conda clean -afy

# Set environment variables
ENV CONDA_DEFAULT_ENV=jax-fem
ENV PATH=/opt/conda/envs/jax-fem/bin:$PATH

# Install PyTorch manually using pip (some PyTorch CUDA wheels are easier this way)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN git clone https://github.com/deepmodeling/jax-fem.git 

RUN pip install optimistix
# Optional: copy other files like requirements.txt or project code
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Default to bash shell with environment activated
CMD ["bash"]
