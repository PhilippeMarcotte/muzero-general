# MuZero Reanalyze Implementation And Investigation

This a reimplementation of MuZero Reanalyze using the open-source version of MuZero Duvaud, Werner; Hainaut, Aurèle and Lenoir, Paul (see [README_MuZeroGeneral](./README_MuZeroGeneral.md)).

The implementation was tested on Cartpole-v1 from OpenAi Gym and the implementation of Tic Tac Toe from the same authors of MuZero General. Two implementations were tested:
- A synchronous one that uses multiple worker to push batches on a queue while updating the target values and policies. The trainer process pulls one batch at a time for training. This implementation stays true to the original descirption in Appendix H of the [original MuZero paper](https://arxiv.org/abs/1911.08265).
- A completely asynchronous one that updates samples directly in the replay buffer. This is much faster but does not faithfully reproduce the process described in the original paper.

## Command for reproducing the results
```shell script
python muzero.py --game_name <configuration name> --action "Train" --logger tensorboard --seed <seed>
```

The configuration used are located in games. However, only the name is required. Here are all the configurations for the experiments:
- basic_tictactoe_ratio_0_5
- true_reanalyze_tictactoe_ratio_0_5
- fast_reanalyze_tictactoe_ratio_0_5
_basic_cartpole_75_ratio_0_25_num_sim