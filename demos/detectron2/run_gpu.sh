#!/usr/bin/env -S bash -e
docker build . -t vistrack-detectron
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-detectron \
           bash
