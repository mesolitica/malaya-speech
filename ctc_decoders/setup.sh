#!/usr/bin/env bash

sudo apt install -y libboost-all-dev swig sox

if [ ! -d kenlm ]; then
    wget https://kheafield.com/code/kenlm.tar.gz
    tar -xzvf kenlm.tar.gz
    echo -e "\n"
fi

if [ ! -d openfst-1.6.3 ]; then
    echo "Download and extract openfst ..."
    wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
    tar -xzvf openfst-1.6.3.tar.gz
    echo -e "\n"
fi

if [ ! -d ThreadPool ]; then
    git clone https://github.com/progschj/ThreadPool.git
    echo -e "\n"
fi

echo "Install decoders ..."
python3 setup.py install --num_processes 4
