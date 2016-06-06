FROM kaixhin/cudnn:latest
MAINTAINER Patrick Callier <pcallier@iqt.org>

# Install python etc.
RUN apt-get update
RUN apt-get install -yq python \
    build-essential \
    python-pip \
    python-virtualenv \
    pkg-config \
    python-dev
RUN apt-get install -yq libhdf5-dev \
    libyaml-dev \
    python-h5py
RUN pip install jupyter \
    filelock

# Install tensorflow
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
RUN python -c 'import tensorflow'

# Install chainer
RUN pip install chainer
RUN python -c 'import chainer'

# Install neon (v1.4.0 release)
RUN apt-get install -yq git
RUN cd /opt && \
    git clone https://github.com/NervanaSystems/neon.git && \
    cd neon && git checkout bc196cb && make sysinstall
RUN python -c 'import neon'

WORKDIR /root

CMD ["/bin/bash"]