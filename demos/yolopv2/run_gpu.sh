#!/usr/bin/env -S bash -e
docker build . -t vistrack-yolopv2
docker run -it --rm \
        --gpus all \
        --shm-size=1gb \
        -v `realpath .`:/demo \
        vistrack-yolopv2 \
        bash