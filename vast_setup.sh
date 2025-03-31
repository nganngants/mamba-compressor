#!/bin/bash

touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

echo "conda activate mambacompressor" >> ~/.bashrc

source ~/.bashrc

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt-get install unzip -y
