# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.134.0/containers/python-3/.devcontainer/base.Dockerfile
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# This Dockerfile adds a non-root 'appuser' user with sudo access. However, for Linux,
# this user's GID/UID must match your local user UID/GID to avoid permission issues
# with bind mounts. Update USER_UID / USER_GID if yours is not 1000. See
# https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=appuser
ARG USER_UID=501
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************




ARG PREFIX=/home/${USERNAME}
ARG BRANCH=v2.0.0-rc1
ARG SHARED=OFF


# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# # [Optional] Allow the vscode user to pip install globally w/o sudo
# ENV PIP_TARGET=/usr/local/pip-global
# ENV PYTHONPATH=${PIP_TARGET}:${PYTHONPATH}
# ENV PATH=${PIP_TARGET}/bin:${PATH}
# RUN if ! cat /etc/group | grep -e "^pip-global:" > /dev/null 2>&1; then groupadd -r pip-global; fi \
#     && usermod -a -G pip-global ${USERNAME} \
#     && umask 0002 && mkdir -p ${PIP_TARGET} \
#     && chown :pip-global ${PIP_TARGET} \
#     && ( [ ! -f "/etc/profile.d/00-restore-env.sh" ] || sed -i -e "s/export PATH=/export PATH=\/usr\/local\/pip-global:/" /etc/profile.d/00-restore-env.sh )

# # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     # [Optional] Add sudo support for the non-root user
#     && export DEBIAN_FRONTEND=noninteractive\
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends apt-utils dialog \
    # OpenImageIO depdendencies.
    && apt-get -y install libboost-all-dev libilmbase-dev libopenexr-dev libpng-dev libtiff5-dev \
    #
    # OpenImageIO.
    && apt-get -y install python3-openimageio \
    #
    # CMake
    && apt-get -y install cmake \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y
    #&& rm -rf /var/lib/apt/lists/*


# Install OpenColorIO
RUN mkdir -p /tmp/build_workspace/build \
    && mkdir -p ${PREFIX} \
    && cd /tmp/build_workspace\
    && git clone --branch ${BRANCH} https://github.com/AcademySoftwareFoundation/OpenColorIO.git \
    && cd /tmp/build_workspace/build && \
    LDFLAGS="-Wl,-rpath,${PREFIX}/lib" \
    CXXFLAGS="-Wno-deprecated-declarations -fPIC" \
        cmake /tmp/build_workspace/OpenColorIO \
            -DCMAKE_INSTALL_PREFIX=$PREFIX \
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
            -DBUILD_SHARED_LIBS=${SHARED} \
    && cd /tmp/build_workspace/build \
    && make install \
    && cd /tmp/build_workspace \
    && rm -rf OpenColorIO \
    && rm -rf build \
    && mkdir -p /workspace \
    && cd /workspace \
    && rm -rf /tmp/build_workspace#


RUN tar -cvf /tmp/ocio_streamlit.tar ${PREFIX}/lib ${PREFIX}/include ${PREFIX}/bin


# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV PYTHONUSERBASE=${PREFIX}

# WORKDIR /workspace

# app root directory
ENV IMAGE_FORMATION_TOOLKIT__ROOT_DIRECTORY=/workspaces/image_formation_toolkit

# # Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=
