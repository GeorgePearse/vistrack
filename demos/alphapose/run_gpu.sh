#!/usr/bin/env -S bash -e
docker build . -t vistrack-alphapose
docker run -it --rm \
           --gpus all \
           --shm-size=5gb \
           -v `realpath .`:/demo \
           vistrack-alphapose \
           bash
