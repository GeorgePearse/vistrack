#!/usr/bin/env -S bash -e
docker build . -t vistrack-yolov5
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-yolov5 \
           bash
