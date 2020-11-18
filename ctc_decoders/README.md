# CTC-Decoder

This beam search decoder implementation was originally taken from Baidu DeepSpech project:
https://github.com/usimarit/ctc_decoders

## Dependencies

```bash
# Mac
brew install boost swig
# Ubuntu
sudo apt install libboost-all-dev swig sox
# Arch
sudo pacman -Syu boost swig sox
```

## Install

```bash
python3 setup.py install
```

## Usage

See [example/decode.py](example/decode.py) for usage.

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