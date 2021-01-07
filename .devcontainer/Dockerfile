#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------
FROM debian:buster

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root 'vscode' user with sudo access. However, for Linux,
# this user's GID/UID must match your local user UID/GID to avoid permission issues
# with bind mounts. Update USER_UID / USER_GID if yours is not 1000. See
# https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install curl git iproute2 lsb-release procps \
    #
    # OpenImageIO depdendencies.
    && apt-get -y install libboost-all-dev libilmbase-dev libopenexr-dev libpng-dev libtiff5-dev \
    #
    # OpenImageIO.
    && apt-get -y install python3-openimageio \
    #
    # CMake
    && apt-get -y install cmake \
    #
    # Pip.
    && apt-get -y install python3-pip \
    #
    # Ninja.
    && pip3 install ninja \
    #
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


# Install OpenColorIO
RUN mkdir -p /tmp/build_workspace/build \
    && cd /tmp/build_workspace \
    && git clone --branch master https://github.com/AcademySoftwareFoundation/OpenColorIO.git \
    && cd build \
    &&  LDFLAGS="-Wl,-rpath,/usr/local/lib" \
        CXXFLAGS="-Wno-deprecated-declarations -fPIC" \
        cmake -G Ninja ../OpenColorIO \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_BUILD_TYPE=Release \
            -DOCIO_BUILD_TESTS=OFF \
            -DOCIO_BUILD_GPU_TESTS=OFF \
            -DOCIO_BUILD_APPS=ON \
            -DOCIO_USE_HEADLESS=ON \
            -DOCIO_WARNING_AS_ERROR=OFF \
            -DOCIO_INSTALL_EXT_PACKAGES=MISSING \
            -DCMAKE_CXX_STANDARD=14 \
            -DOCIO_BUILD_PYTHON=ON \
            -DOCIO_BUILD_JAVA=OFF \
            -DBUILD_SHARED_LIBS=OFF \
    && ninja install \
    && cd /tmp/build_workspace \
    && rm -rf OpenColorIO \
    && rm -rf build

RUN mkdir /workspace
WORKDIR /workspace

RUN if [ -f "requirements.txt" ]; then pip3 install -r requirements.txt && rm requirements.txt*; fi

# app root directory
ENV IMAGE_FORMATION_TOOLKIT__ROOT_DIRECTORY=/workspaces/image_formation_toolkit

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

# Set the default shell to bash rather than sh
ENV SHELL /bin/bash