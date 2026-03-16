
FROM phd_orbslam:latest

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
    colcon-core \
    colcon-common-extensions\
    setuptools\
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
    setuptools \
    transforms3d 







