#!/usr/bin/env bash
docker build . -t vistrack-mmdetection
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           vistrack-mmdetection \
           bash
