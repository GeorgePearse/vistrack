#!/usr/bin/env -S bash -e
docker build . -t vistrack-yolov4
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           vistrack-yolov4 \
           bash
