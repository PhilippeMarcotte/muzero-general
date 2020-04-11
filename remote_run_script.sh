#!/usr/bin/env bash
sudo mount /dev/disk/by-uuid/9dc84c5a-2407-4f99-bf07-0530ef1d21fd /srv/data
cd /srv/data/muzero-general || sudo shutdown -h now

git checkout dev
git pull

tsp experiments.sh