FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install system dependencies and libraries
RUN apt-get update -qqy && \
    apt-get -qqy upgrade && \
    apt-get install -qqy --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        coreutils \
        curl \
        environment-modules \
        gfortran \
        git \
        gnupg \
        gnupg2 \
        gzip \
        libatlas-base-dev \
        libblas-dev \
        libboost-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libcurl4-gnutls-dev \
        libexpat1-dev \
        libfreetype6-dev \
        libgeos-dev \
        libgif-dev \
        libhdf5-dev \
        libjpeg-dev \
        liblapack-dev \
        liblzma-dev \
        libnetcdf-dev \
        libopenblas-dev \
        libopencv-dev \
        libpng-dev \
        libpoppler-dev \
        libpq-dev \
        libproj-dev \
        libspatialite-dev \
        libsqlite3-dev \
        libtiff-dev \
        libxerces-c-dev \
        libzstd-dev \
        lsb-release \
        patch \
        pkg-config \
        tar \
        tmux \
        unzip \
        zip && \
    apt-get clean -qqy && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
ENV UV_INSTALL_DIR=/usr/local/bin
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && \
    rm /install.sh

RUN curl -fsSL https://claude.ai/install.sh | bash

# Always activate virtual env when opening a new terminal
RUN echo 'source .venv/bin/activate' >> /home/vscode/.bashrc