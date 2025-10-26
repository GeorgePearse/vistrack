#!/usr/bin/env -S bash -e
docker build . -t vistrack-trt-profiling
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           vistrack-trt-profiling \
           bash
