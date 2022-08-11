#!/bin/bash

export CUDA_HOME=/usr/local/cuda-11.2

if [ ! -d warp-transducer ]; then
    git clone https://github.com/huseinzol05/warp-transducer.git
fi
cd ./warp-transducer
mkdir -p build && cd build
cmake \
-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" ..

make
cd ../tensorflow_binding

# I use virtual env
PYTHONPATH=~/tf-nvidia/lib/python3.8/site-packages ~/tf-nvidia/bin/python3 setup.py install

wget -qO-  https://raw.githubusercontent.com/huseinzol05/warp-transducer/master/tensorflow_binding/tests/test_warprnnt_op.py | ~/tf-nvidia/bin/python3 -