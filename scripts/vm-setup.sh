#!/bin/bash

# install docker
curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh

# setup docker permissions
sudo groupadd docker
sudo chmod 666 /var/run/docker.sock

# install cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

# update cuda
chmod a+x scripts/cuda-11.8.sh
/bin/bash scripts/cuda-11.8.sh

# update nvidia-docker
sudo apt install nvidia-docker2