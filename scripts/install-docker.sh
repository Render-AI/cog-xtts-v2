#!/bin/bash

# install docker
curl -fsSL get.docker.com -o get-docker.sh && sh get-docker.sh

# setup docker permissions
sudo groupadd docker
sudo chmod 666 /var/run/docker.sock