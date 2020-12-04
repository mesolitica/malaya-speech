#!/bin/bash

export CUDA_HOME=/usr/local/cuda-10.0

ln -s /usr/local/cuda/lib64/libcurand.so.10.0 /usr/local/cuda/lib64/libcurand.so.10
mkdir -p externals && cd externals

if [ ! -d warp-transducer ]; then
    git clone https://github.com/noahchalifour/warp-transducer.git
fi
cd ./warp-transducer
mkdir -p build && cd build
cmake \
-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" ..

make
cd ../tensorflow_binding
python3 setup.py install --user

wget -qO-  https://raw.githubusercontent.com/usimarit/warp-transducer/master/tensorflow_binding/tests/test_basic.py | python3 -