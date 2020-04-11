#!/usr/bin/env bash
games=("basic_cartpole" "per_cartpole" "reanalyze_cartpole" "reanalyze_per_cartpole")
action="Train"
seeds=(0 10 20 30 40)

for seed in "${seeds[@]}"; do
  for game in "${games[@]}"; do
    python muzero.py --game_name "$game" --action $action --seed "$seed" --group "$game"
  done
done

sudo shutdown -h now
