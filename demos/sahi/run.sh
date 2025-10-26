#!/usr/bin/env -S bash -e
docker build . -t vistrack-sahi
docker run -it --rm \
           --gpus all \
           --shm-size=5gb \
           -v `realpath .`:/demo \
           vistrack-sahi \
           bash
