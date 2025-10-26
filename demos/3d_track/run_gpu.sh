#!/usr/bin/env -S bash -e
docker build . -t vistrack-3d
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           vistrack-3d \
           bash
