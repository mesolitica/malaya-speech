# CTC-Decoder

This beam search decoder implementation was originally taken from Baidu DeepSpech project:
https://github.com/usimarit/ctc_decoders

## Install

### From PYPI

```bash
pip3 install ctc-decoders
```

If for linux, simply download specific python version at [Available whl](#Available-whl).

### From source

1. Install dependencies,

For mac,

```bash
brew install boost swig
```

For Ubuntu / Debian,

```bash
sudo apt install libboost-all-dev swig sox
```

For Arch,

```bash
sudo pacman -Syu boost swig sox
```

2. Compile using [setup.py](setup.py),

```bash
python3 setup.py install
python3 setup.py install --num_processes 4
```

Or build to whl,

```
python3 setup.py sdist bdist_wheel
```

Or for manylinux whl,

```bash
PLAT=manylinux1_x86_64
DOCKER_IMAGE=quay.io/pypa/manylinux1_x86_64
docker run --rm -e PLAT=$PLAT -v `pwd`:/io $DOCKER_IMAGE $PRE_CMD /io/build-wheel.sh
```


## Available whl

1. [ctc_decoders-1.0-cp37-cp37m-macosx_10_9_x86_64.whl](https://f000.backblazeb2.com/file/malaya-speech-model/ctc-decoder/ctc_decoders-1.0-cp37-cp37m-macosx_10_9_x86_64.whl)
2. [ctc_decoders-1.0-cp36-cp36m-linux_x86_64.whl](https://f000.backblazeb2.com/file/malaya-speech-model/ctc-decoder/ctc_decoders-1.0-cp36-cp36m-linux_x86_64.whl)
3. [ctc_decoders-1.0-cp37-cp37m-linux_x86_64.whl](https://f000.backblazeb2.com/file/malaya-speech-model/ctc-decoder/ctc_decoders-1.0-cp37-cp37m-linux_x86_64.whl)
4. [ctc_decoders-1.0-cp38-cp38-linux_x86_64.whl](https://f000.backblazeb2.com/file/malaya-speech-model/ctc-decoder/ctc_decoders-1.0-cp38-cp38-linux_x86_64.whl)

## Usage

See [example/test.py](example/test.py) for usage.

## License

```
Copyright 2019 Baidu, Inc.
Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```