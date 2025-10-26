#!/usr/bin/env bash
docker build . -t vistrack-openpose
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-openpose \
           bash
