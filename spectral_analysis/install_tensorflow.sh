#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    # And just grab the wheel for tensorflow
    pip3 install tensorflow==2.7.0 -f https://tf.kmtea.eu/whl/stable.html
else
    pip3 install tensorflow==2.7.0
    pip3 install tensorflow-io==0.22.0
    pip3 install tensorflow-io-gcs-filesystem==0.22.0
fi
