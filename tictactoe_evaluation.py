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
from abc import ABC, abstractmethod
import copy


class TictactoeComp(ABC):
    def __init__(self, me, other):
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        self.me = me
        self.other = other

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

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

    @staticmethod
    def coord_to_index(x, y):
        row = np.arange(3)
        column = np.arange(3)
        return row[x] * 3 + column[y]

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

    def game_over(self, state):
        """
        This function test if the human or computer wins
        :param state: the state of the current board
        :return: True if the human or computer wins
        """
        return self.wins(state, self.other) or self.wins(state, self.me)

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


class Random(TictactoeComp):
    def __call__(self, state, depth, player):
        empty_cells = self.empty_cells(state)
        cell = np.random.randint(0, len(empty_cells))
        return self.coord_to_index(*empty_cells[cell])


class Intermediate(TictactoeComp):
    def __call__(self, state, depth, player):
        legal_moves = self.empty_cells(state)
        for move in legal_moves:
            x, y = move
            new_state = copy.deepcopy(state)
            new_state[x, y] = player
            if self.wins(new_state, player):
                return self.coord_to_index(x, y)

        opponent = -player
        for move in legal_moves:
            x, y = move
            new_state = copy.deepcopy(state)
            new_state[x, y] = opponent
            if self.wins(new_state, opponent):
                return self.coord_to_index(x, y)

        cell = np.random.randint(0, len(legal_moves))
        return self.coord_to_index(*legal_moves[cell])


class Expert(TictactoeComp):
    def __call__(self, state, depth, player):
        if (np.array(state) == 0).all():
            return np.random.choice([0, 2, 6, 8])
        x, y, player = self.minmax(state, depth, player)
        return self.coord_to_index(x, y)

    def minmax(self, state, depth, player):
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
            score = self.minmax(state, depth - 1, -player)
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
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    game_history1 = GameHistory()
    game_history1.action_history.append(0)
    game_history1.reward_history.append(0)
    game_history1.to_play_history.append(game.to_play())
    game_history1.legal_actions.append(game.legal_actions())
    observation1 = copy.deepcopy(observation)
    # observation1[0] = -observation1[1]
    # observation1[1] = -observation1[0]
    # observation1[2] = -observation1[2]
    game_history1.observation_history.append(observation1)

    game_history2 = GameHistory()
    game_history2.action_history.append(0)
    game_history2.reward_history.append(0)
    game_history2.to_play_history.append(not game.to_play())
    game_history2.legal_actions.append(game.legal_actions())
    observation2 = copy.deepcopy(observation)
    observation2[0] = -observation2[1]
    observation2[1] = -observation2[0]
    observation2[2] = -observation2[2]
    game_history2.observation_history.append(observation2)

    done = False
    reward = 0

    while not done:
        if game.to_play_real() == 1:
            config = config1
            model = model1
            game_history = game_history1
        else:
            config = config2
            model = model2
            game_history = game_history2

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

        game_history1.store_search_statistics(root, config.action_space)
        game_history1.priorities.append(priority)
        game_history2.store_search_statistics(root, config.action_space)
        game_history2.priorities.append(priority)
        observation, reward, done = game.step(action)
        if render:
            game.render()

        game_history1.action_history.append(action)
        observation1 = copy.deepcopy(observation)
        # observation1[0] = -observation1[1]
        # observation1[1] = -observation1[0]
        # observation1[2] = -observation1[2]
        game_history1.observation_history.append(observation1)
        game_history1.reward_history.append(reward)
        game_history1.to_play_history.append(game.to_play())
        game_history1.legal_actions.append(game.legal_actions())

        game_history2.action_history.append(action)
        observation2 = copy.deepcopy(observation)
        observation2[0] = -observation2[1]
        observation2[1] = -observation2[0]
        observation2[2] = -observation2[2]
        game_history2.observation_history.append(observation2)
        game_history2.reward_history.append(reward)
        game_history2.to_play_history.append(not game.to_play())
        game_history2.legal_actions.append(game.legal_actions())

    return reward, TictactoeComp.wins(game.get_state(), 1)


def _play_against_algorithm(args):
    return play_against_algorithm(*args)


def evaluate_against_other(weights1, config1, weights2, config2, n_tests=20, render=False, seed=0):
    player1_win = 0
    player2_win = 0
    draw = 0

    if render:
        reward, player1_won = play_against_other(weights1, config1, weights2, config2, seed, render=render)
        if reward:
            if player1_won:
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


def play_against_algorithm(weight_file_path, config_name, seed, algo="expert", render=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    game_module = importlib.import_module("games." + config_name)
    config = game_module.MuZeroConfig()
    model = models.MuZeroNetwork(config)
    model.set_weights(torch.load(weight_file_path))
    model.eval()

    algo = globals()[algo.capitalize()](-1, 1)

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
        if game.to_play_real() == -1:
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
        if render:
            game.render()
        depth -= 1

        game_history.action_history.append(action)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(reward)
        game_history.to_play_history.append(game.to_play())
        game_history.legal_actions.append(game.legal_actions())

    return reward, TictactoeComp.wins(game.get_state(), 1)


def evaluate_against_algorithm(weights, config, algorithm="expert", n_tests=20, render=False, seed=0):
    player1_win = 0
    player2_win = 0
    draw = 0

    if render:
        reward, player1_won = play_against_algorithm(weights, config, seed, algorithm, render=True)
        if reward:
            if player1_won:
                player1_win += 1
            else:
                player2_win += 1
        else:
            draw += 1
    else:
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
    fire.Fire({"pve": evaluate_against_algorithm,
               "pvp": evaluate_against_other})
