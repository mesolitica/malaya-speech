FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime as base

RUN apt update
RUN apt install git ffmpeg -y
ADD ./requirements.txt .
RUN pip3 install -r requirements.txt
RUN apt update
RUN apt install build-essential -y
RUN pip3 install torchvision==0.21.0
RUN pip3 install gradio==5.37.0 gradio_client==1.10.4
ADD ./ .