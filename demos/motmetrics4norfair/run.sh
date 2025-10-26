#!/usr/bin/env -S bash -e
docker build . -t vistrack-motmetrics
docker run -it --rm \
           -v `realpath .`:/demo \
           vistrack-motmetrics \
           bash
