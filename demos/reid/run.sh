#!/usr/bin/env -S bash -e
docker build . -t vistrack-reid
docker run -it --rm \
           -v `realpath .`:/demo \
           vistrack-reid \
           bash
