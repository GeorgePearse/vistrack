#!/usr/bin/env -S bash -e
docker build . -t vistrack-yolov7
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-yolov7 \
           bash
