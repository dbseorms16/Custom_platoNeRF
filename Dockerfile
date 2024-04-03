FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 
RUN apt-get clean 
RUN rm -rf /var/lib/apt/lists/*
apt-get install libgl1-mesa-glx


RUN echo code --install-extension eamodio.gitlens 
RUN echo code --install-extension formulahendry.terminal