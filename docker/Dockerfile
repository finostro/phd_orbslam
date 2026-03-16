FROM ros:rolling-ros-core AS base

# install ros package
RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      libboost-all-dev \
      libeigen3-dev \
      libopencv-dev \
      libg2o-dev \
      ros-${ROS_DISTRO}-demo-nodes-cpp \
      ros-${ROS_DISTRO}-demo-nodes-py \
      ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
      ros-${ROS_DISTRO}-tf2 \
      ros-${ROS_DISTRO}-sophus \
      ros-${ROS_DISTRO}-gtsam \
      libyaml-cpp-dev \
      && \
    rm -rf /var/lib/apt/lists/*


CMD ["bash"]


FROM base AS finostro_dev

ARG UNAME=finostro
ARG UID=1000
ARG GID=1000

RUN set -eux; \
    : "${UID:?UID not set}" "${GID:?GID not set}" "${UNAME:?UNAME not set}"; \
    \
    # Ensure group with GID exists and has the correct name
    if getent group "${GID}" >/dev/null; then \
        EXISTING_GROUP="$(getent group "${GID}" | cut -d: -f1)"; \
        if [ "${EXISTING_GROUP}" != "${UNAME}" ]; then \
            groupmod -n "${UNAME}" "${EXISTING_GROUP}"; \
        fi; \
    else \
        if getent group "${UNAME}" >/dev/null; then \
            groupmod -g "${GID}" "${UNAME}"; \
        else \
            groupadd -g "${GID}" "${UNAME}"; \
        fi; \
    fi; \
    \
    # Ensure user with UID exists and has correct name, group, and home
    if getent passwd "${UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${UID}" | cut -d: -f1)"; \
        if [ "${EXISTING_USER}" != "${UNAME}" ]; then \
            usermod -l "${UNAME}" "${EXISTING_USER}"; \
        fi; \
        usermod -g "${GID}" -d "/home/${UNAME}" -m "${UNAME}"; \
    else \
        if getent passwd "${UNAME}" >/dev/null; then \
            usermod -u "${UID}" -g "${GID}" -d "/home/${UNAME}" -m "${UNAME}"; \
        else \
            useradd -m -u "${UID}" -g "${GID}" -d "/home/${UNAME}" "${UNAME}"; \
        fi; \
    fi; \
    passwd -d "${UNAME}"

# #FIX Permisions in unterlying image
# RUN chmod -R a+rX /usr/local/lib/python3.12/dist-packages && \
#  chmod -R a+rX /root && \
#  chmod -R a+rX /usr/local 

RUN apt-get update && apt-get install -y \
      git \
      ros-${ROS_DISTRO}-rviz2 \
      sudo \
       && \
    rm -rf /var/lib/apt/lists/*
  

USER $UNAME

WORKDIR /home/${UNAME}/ws


RUN cd \
  && git clone https://github.com/finostro/dotfiles.git \ 
  && cd dotfiles \
  && ./install.sh \
  && . ${HOME}/venv/bin/activate \
    && pip3 install -U --force-reinstall \
    colcon-common-extensions\
    colcon-core \
    setuptools \
    && pip3 install -U  \
    Shapely \
    cmake \
    colcon-common-extensions \
    colcon-core \
    geopandas \
    numpy \
    packaging \
    pip \
    pyquaternion \
    setuptools\
    transforms3d 







