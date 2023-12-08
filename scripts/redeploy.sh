#!/bin/bash

# git clone
sudo rm -rf cog-xtts-v2
git clone https://github.com/render-ai/cog-xtts-v2
sudo groupadd docker
sudo chmod 666 /var/run/docker.sock
sudo systemctl start docker
bash cog-xtts-v2/scripts/deploy.sh