#!/usr/bin/env -S bash -e
docker build . -t vistrack-camera-motion
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-camera-motion \
           bash
