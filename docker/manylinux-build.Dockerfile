FROM quay.io/pypa/manylinux2014_x86_64:latest

ARG PACKAGE_VERSION

ENV PATH="/opt/python/cp39-cp39/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN yum install -y cmake ninja-build && yum clean all

RUN python -m pip install --upgrade pip && \
    pip install build setuptools_scm

WORKDIR /workspace
