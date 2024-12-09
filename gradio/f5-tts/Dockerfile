FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime as base

RUN apt update
RUN apt install git ffmpeg -y
ADD ./requirements.txt .
RUN pip3 install -r requirements.txt
ADD ./ .