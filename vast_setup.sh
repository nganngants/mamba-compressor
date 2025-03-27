#!/bin/bash

touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

echo "conda activate mambacompressor" >> ~/.bashrc

source ~/.bashrc

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt-get install unzip -y

# conda install -c nvidia cuda-compiler -y
# conda activate mambacompressor
pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
pip install deepspeed
pip install h5py
pip install flash-attn --no-build-isolation
pip uninstall deepspeed -y
pip install deepspeed==0.15.4
pip install gdown
pip install -e .

git config --global user.email "nganngants@gmail.com"
git config --global user.name "nganngants"

gdown --folder https://drive.google.com/drive/folders/17f8lbKdSSN7A3gaZkIvgAVKLnlCNqDI7?usp=drive_link
gdown --folder https://drive.google.com/drive/folders/1-CYUhPkADMc4I5sycGrT7TP78iTU87wN\?usp\=drive_link
gdown 17nHhoVx8im-fjsY7cqFL2OrMwVusHMHJ

unzip mamba_ckpt.zip

git submodule update --init
cd VideoLLaMA2
git checkout audio_visual
pip install -e .
cd ..