FROM ubuntu:16.04
RUN ls
RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1
ENV CODE_DIR /root/code
ENV VENV /root/venv
RUN pip install --upgrade pip
RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    mkdir $CODE_DIR && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    pip install codacy-coverage && \
    pip install scipy && \
    pip install joblib && \
    pip install mpi4py && \
    pip install cloudpickle && \
    pip install tensorflow==2.0.0 && \
    pip install numpy && \
    pip install pandas && \
    pip install pytest && \
    pip install pytest-cov && \
    pip install pytest-env && \
    pip install pytest-xdist && \
    pip install matplotlib && \
    pip install gym

ENV PATH=$VENV/bin:$PATH

RUN apt-get -y update && apt-get -y install curl jq

RUN python -m pytest -s -v tests matd3
