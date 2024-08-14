#! /bin/bash
workdir=/home/linaro/workspace/PixArt-sigma/scripts
python3 ${workdir}/inference_np.py &
python3 ${workdir}/server.py &