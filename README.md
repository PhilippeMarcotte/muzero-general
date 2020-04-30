# MuZero Reanalyze Implementation And Investigation

This a reimplementation of MuZero Reanalyze using the open-source version of MuZero Duvaud, Werner; Hainaut, Aurèle and Lenoir, Paul (see [README_MuZeroGeneral](./README_MuZeroGeneral.md)).

The implementation was tested on Cartpole-v1 from OpenAi Gym and the implementation of Tic Tac Toe from the same authors of MuZero General. Two implementations were tested:
- A synchronous one that uses multiple worker to push batches on a queue while updating the target values and policies. The trainer process pulls one batch at a time for training. This implementation stays true to the original descirption in Appendix H of the [original MuZero paper](https://arxiv.org/abs/1911.08265).
- A completely asynchronous one that updates samples directly in the replay buffer. This is much faster but does not faithfully reproduce the process described in the original paper.

## Installation
```shell script
git clone https://github.com/PhilippeMarcotte/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

## Command for reproducing the results
```shell script
python muzero.py --game_name <configuration name> --action "Train" --logger tensorboard --seed <seed>
```

The configuration used are located in the [games](./games) fodler. However, only the name is required. Here are all the configurations used for the experiments:
- basic_tictactoe_ratio_0_5
- true_reanalyze_tictactoe_ratio_0_5
- fast_reanalyze_tictactoe_ratio_0_5

- basic_cartpole_75_ratio_0_25 (seed=[0,10,20,30,40])
- basic_cartpole_75_ratio_0_5 (seed=[0,10,20,30,40])
- basic_cartpole_75_ratio_1 (seed=[0,10,20,30,40])
- basic_cartpole_75_ratio_2 (seed=[0,10,20,30,40])

- true_reanalyze_cartpole_75_ratio_0_25 (seed=[0,10,20,50,60])
- true_reanalyzebasic_cartpole_75_ratio_0_5 (seed=[0,10,20,50,60])
- true_reanalyze_cartpole_75_ratio_1 (seed=[0,10,20,50,60])
- true_reanalyze_cartpole_75_ratio_2 (seed=[0,10,20,50,60])

- fast_reanalyze_cartpole_75_ratio_0_25 (seed=[0,10,20,30,40])
- fast_reanalyze_cartpole_75_ratio_0_5 (seed=[0,10,20,30,40])
- fast_reanalyze_cartpole_75_ratio_1 (seed=[0,10,20,30,40])
- fast_reanalyze_cartpole_75_ratio_2 (seed=[0,10,20,30,40])

For further information please see the original [README](./README_MuZeroGeneral.md).
