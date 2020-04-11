#!/usr/bin/env bash
sudo mount /dev/disk/by-uuid/9dc84c5a-2407-4f99-bf07-0530ef1d21fd /srv/data
cd /srv/data/muzero-general || exit

git checkout dev
git pull

games=("basic_cartpole" "per_cartpole" "reanalyze_cartpole" "reanalyze_per_cartpole")
action="Train"
seeds=(0 10 20 30 40)

for seed in "${seeds[@]}"; do
  for game in "${games[@]}"; do
    python muzero.py --game_name "$game" --action $action --seed "$seed" --group "$game"
  done
done
