#!/usr/bin/env -S bash -e
docker build . -t vistrack-bbx-kp
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-bbx-kp \
           bash
