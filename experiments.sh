#!/usr/bin/env bash
games=("true_reanalyze_cartpole")
action="Train"
seeds=(0 10 20 50 60)
bufferSizes=(100 75 50 25)
ratios=("0_5" "1" "2")

for ratio in "${ratios[@]}"; do
  for game in "${games[@]}"; do
    for seed in "${seeds[@]}"; do
      python muzero.py --game_name "${game}_75_ratio_${ratio}_sim_50" --action $action --seed "$seed" --group "$game" --tags "['buffer_size=75', 'ratio=${ratio/_/.}']"
    done
  done
done
