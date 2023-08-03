#!/bin/bash
set -e

UNAME=`uname -m`
BUILD_WHEEL=0

# for other platforms we can download prebuilt wheels from pypi, so no need to build
# they'll get installed from requirements-blocks.txt later in the Dockerfile
if [ "$UNAME" == "aarch64" ]; then
    if [ "$BUILD_WHEEL" -eq "0" ]; then
        wget https://cdn.edgeimpulse.com/build-system/wheels/aarch64/jaxlib-0.4.1-cp38-cp38-manylinux2014_aarch64.whl
        pip3 install jaxlib-0.4.1-cp38-cp38-manylinux2014_aarch64.whl
        rm jaxlib-0.4.1-cp38-cp38-manylinux2014_aarch64.whl
    else
        apt update && apt install -y g++ python python3-dev git
        pip3 install numpy==1.21.0 wheel
        git clone --branch jaxlib-v0.4.1 https://github.com/google/jax
        cd jax
        python3 build/build.py
        pip3 install dist/*.whl  # installs jaxlib (includes XLA)
        cd ..
        rm -r jax
    fi
fi
