from math import inf as infinity
from games.tictactoe import Game
import fire
import numpy as np
from multiprocessing import Pool
from muzero import MuZero
import models
import importlib
import torch
from self_play import MCTS, SelfPlay, GameHistory
import tqdm


class Expert:
    def __init__(self, me, other):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        self.me = me
        self.other = other

    def evaluate(self, state):
        """
        Function to heuristic evaluation of state.
        :param state: the state of the current board
        :return: +1 if the computer wins; -1 if the human wins; 0 draw
        """
        if self.wins(state, self.me):
            score = +1
        elif self.wins(state, self.other):
            score = -1
        else:
            score = 0

        return score

    @staticmethod
    def wins(state, player):
        """
        This function tests if a specific player wins. Possibilities:
        * Three rows    [X X X] or [O O O]
        * Three cols    [X X X] or [O O O]
        * Two diagonals [X X X] or [O O O]
        :param state: the state of the current board
        :param player: a human or a computer
        :return: True if the player wins
        """
        win_state = [
            [state[0][0], state[0][1], state[0][2]],
            [state[1][0], state[1][1], state[1][2]],
            [state[2][0], state[2][1], state[2][2]],
            [state[0][0], state[1][0], state[2][0]],
            [state[0][1], state[1][1], state[2][1]],
            [state[0][2], state[1][2], state[2][2]],
            [state[0][0], state[1][1], state[2][2]],
            [state[2][0], state[1][1], state[0][2]],
        ]
        if [player, player, player] in win_state:
            return True
        else:
            return False

    def game_over(self, state):
        """
        This function test if the human or computer wins
        :param state: the state of the current board
        :return: True if the human or computer wins
        """
        return self.wins(state, self.other) or self.wins(state, self.me)

    @staticmethod
    def empty_cells(state):
        """
        Each empty cell will be added into cells' list
        :param state: the state of the current board
        :return: a list of empty cells
        """
        cells = []

        for x, row in enumerate(state):
            for y, cell in enumerate(row):
                if cell == 0:
                    cells.append([x, y])

        return cells

    def valid_move(self, x, y):
        """
        A move is valid if the chosen cell is empty
        :param x: X coordinate
        :param y: Y coordinate
        :return: True if the board[x][y] is empty
        """
        if [x, y] in self.empty_cells(self.board):
            return True
        else:
            return False

    def set_move(self, x, y, player):
        """
        Set the move on board, if the coordinates are valid
        :param x: X coordinate
        :param y: Y coordinate
        :param player: the current player
        """
        if self.valid_move(x, y):
            self.board[x][y] = player
            return True
        else:
            return False

    def __call__(self, state, depth, player):
        x, y, player = self.minimax(state, depth, player)
        row = np.arange(len(self.board))
        column = np.arange(len(self.board[0]))
        return row[x] * 3 + column[y]

    def minimax(self, state, depth, player):
        """
        AI function that choice the best move
        :param state: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9),
        but never nine in this case (see iaturn() function)
        :param player: an human or a computer
        :return: a list with [the best row, best col, best score]
        """
        if player == self.me:
            best = [-1, -1, -infinity]
        else:
            best = [-1, -1, +infinity]

        if depth == 0 or self.game_over(state):
            score = self.evaluate(state)
            return [-1, -1, score]

        for cell in self.empty_cells(state):
            x, y = cell[0], cell[1]
            state[x][y] = player
            score = self.minimax(state, depth - 1, -player)
            state[x][y] = 0
            score[0], score[1] = x, y

            if player == self.me:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        return best


def _play_against_other(args):
    return play_against_other(*args)


def play_against_other(weights1, config1, weights2, config2, seed, render=False):
    game_module = importlib.import_module("games." + config1)
    config1 = game_module.MuZeroConfig()
    model1 = models.MuZeroNetwork(config1)
    model1.set_weights(torch.load(weights1))
    model1.eval()

    game_module = importlib.import_module("games." + config2)
    config2 = game_module.MuZeroConfig()
    model2 = models.MuZeroNetwork(config2)
    model2.set_weights(torch.load(weights2))
    model2.eval()

    game = Game(seed)
    observation = game.reset()

    game_history = GameHistory()
    game_history.action_history.append(0)
    game_history.reward_history.append(0)
    game_history.to_play_history.append(game.to_play())
    game_history.legal_actions.append(game.legal_actions())
    game_history.observation_history.append(observation)

    done = False
    reward = 0

    while not done:
        if game.to_play_real() == 1:
            config = config1
            model = model1
        else:
            config = config2
            model = model2

        stacked_observations = game_history.get_stacked_observations(
            -1, config.stacked_observations,
        )

        root, priority, tree_depth = MCTS(config).run(
            model,
            stacked_observations,
            game.legal_actions(),
            game.to_play(),
            False,
        )

        action = SelfPlay.select_action(
            root,
            0,
        )

        game_history.store_search_statistics(root, config.action_space)
        game_history.priorities.append(priority)
        observation, reward, done = game.step(action)
        if render:
            game.render()

        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        game_history.to_play_history.append(game.to_play())
        game_history.legal_actions.append(game.legal_actions())

    return reward, Expert.wins(game.get_state(), 1)


def _play_against_algorithm(args):
    return play_against_algorithm(*args)


def evaluate_against_other(weights1, config1, weights2, config2, n_tests=20, render=False, seed=0):
    player1_win = 0
    player2_win = 0
    draw = 0

    if render:
        reward, player = play_against_other(weights1, config1, weights2, config2, seed, render=render)
        if reward:
            if player == 1:
                player1_win += 1
            else:
                player2_win += 1
        else:
            draw += 1
    else:
        pool = Pool()
        for reward, player1_won in tqdm.tqdm(pool.imap(_play_against_other,
                                                       zip([weights1] * n_tests, [config1] * n_tests,
                                                           [weights2] * n_tests, [config2] * n_tests,
                                                           np.arange(n_tests))), total=n_tests):
            if reward:
                if player1_won:
                    player1_win += 1
                else:
                    player2_win += 1
            else:
                draw += 1

    print(player1_win, player2_win, draw)


def play_against_algorithm(weight_file_path, config_name, seed, algo="expert"):
    game_module = importlib.import_module("games." + config_name)
    config = game_module.MuZeroConfig()
    model = models.MuZeroNetwork(config)
    model.set_weights(torch.load(weight_file_path))
    model.eval()

    if algo == "expert":
        algo = Expert(-1, 1)

    game = Game(seed)
    observation = game.reset()

    game_history = GameHistory()
    game_history.action_history.append(0)
    game_history.reward_history.append(0)
    game_history.to_play_history.append(game.to_play())
    game_history.legal_actions.append(game.legal_actions())
    game_history.observation_history.append(observation)

    done = False
    depth = 9
    reward = 0

    while not done:
        if game.to_play_real() == 1:
            action = algo(game.get_state(), depth, game.to_play_real())
        else:
            stacked_observations = game_history.get_stacked_observations(
                -1, config.stacked_observations,
            )

            root, priority, tree_depth = MCTS(config).run(
                model,
                stacked_observations,
                game.legal_actions(),
                game.to_play(),
                False,
            )

            action = SelfPlay.select_action(
                root,
                0,
            )

            game_history.store_search_statistics(root, config.action_space)
            game_history.priorities.append(priority)
        observation, reward, done = game.step(action)
        depth -= 1

        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        game_history.to_play_history.append(game.to_play())
        game_history.legal_actions.append(game.legal_actions())

    return reward, Expert.wins(game.get_state(), 1)


def evaluate_against_algorithm(weights, config, algorithm="expert", n_tests=20):
    player1_win = 0
    player2_win = 0
    draw = 0

    pool = Pool()

    for reward, player1_won in tqdm.tqdm(
            pool.imap(_play_against_algorithm,
                      zip([weights] * n_tests, [config] * n_tests, np.arange(n_tests), [algorithm] * n_tests)),
            total=n_tests):
        if reward:
            if player1_won:
                player1_win += 1
            else:
                player2_win += 1
        else:
            draw += 1

    print(player1_win, player2_win, draw)


if __name__ == "__main__":
    fire.Fire(evaluate_against_algorithm)
