#! /bin/bash
workdir=/home/linaro/workspace/PixArt-sigma/scripts
pkill -f "python3 ${workdir}/inference_np.py"
pkill -f "python3 ${workdir}/server.py"