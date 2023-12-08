#!/bin/bash

# cog push
cd cog-xtts-v2
sudo groupadd docker
sudo chmod 666 /var/run/docker.sock
sudo systemctl start docker
cog push r8.im/platform-kit/xtts