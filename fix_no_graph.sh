#!/bin/bash

apt-get update && apt-get install -y nvidia-cuda-toolkit

apt-get install -y libnvrtc11.2 libnvrtc-builtins11.5
ln -s /usr/lib/x86_64-linux-gnu/libnvrtc.so.11.2 /usr/lib/x86_64-linux-gnu/libnvrtc.so.12
ln -s /usr/lib/x86_64-linux-gnu/libnvrtc.so.11.2 /usr/lib/x86_64-linux-gnu/libnvrtc.so