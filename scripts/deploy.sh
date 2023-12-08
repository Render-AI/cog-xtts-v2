#!/bin/bash

# cog push
cd cog-xtts-v2
bash cog-xtts-v2/scripts/setup-docker.sh
bash cog-xtts-v2/scripts/start-docker.sh
cog push r8.im/platform-kit/xtts