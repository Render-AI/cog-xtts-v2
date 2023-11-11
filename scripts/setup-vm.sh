#!/bin/bash

# install docker
bash scripts/install-docker.sh

# install nvidia-docker
bash scripts/install-nvidia-docker.sh

# install cuda
bash scripts/install-cuda.sh

# setup docker permissions
sudo groupadd docker
sudo chmod 666 /var/run/docker.sock

# install cog
bash scripts/install-cog.sh