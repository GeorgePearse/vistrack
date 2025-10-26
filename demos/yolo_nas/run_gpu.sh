#!/usr/bin/env -S bash -e
docker build . -t vistrack-yolonas
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-yolonas \
           bash
