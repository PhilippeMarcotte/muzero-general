import datetime
import os

import gym
import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (3, 3,
                                  3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(9)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(2)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 9  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = 6  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels = 16  # Number of channels before heads of dynamic and prediction networks
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 20000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        # Muzero Reanalyze
        self.policy_update_rate = 0.8

        ### Replay Buffer
        self.window_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = False  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = False  # Use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 0.5  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TicTacToe(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def to_play_real(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play_real()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def encode_board(self):
        return self.env.encode_board()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        "Enter the row (1, 2 or 3) to play for the player {}: ".format(
                            self.to_play()
                        )
                    )
                )
                col = int(
                    input(
                        "Enter the column (1, 2 or 3) to play for the player {}: ".format(
                            self.to_play()
                        )
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                        choice in self.legal_actions()
                        and 1 <= row
                        and 1 <= col
                        and row <= 3
                        and col <= 3
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = 3 - action_number // 3
        col = action_number % 3 + 1
        return "Play row {}, column {}".format(row, col)

    def get_state(self):
        return self.env.board


class TicTacToe:
    def __init__(self, seed):
        numpy.random.seed(seed)
        self.board = numpy.zeros((3, 3)).astype(int)
        self.player = numpy.random.choice([-1, 1])

    def to_play(self):
        return 0 if self.player == 1 else 1

    def to_play_real(self):
        return self.player

    def reset(self):
        self.board = numpy.zeros((3, 3)).astype(int)
        self.player = numpy.random.choice([-1, 1])
        return self.get_observation()

    def step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player

        done = self.is_finished()

        reward = 1 if done and 0 < len(self.legal_actions()) else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((3, 3), self.player).astype(float)
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def is_finished(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3).astype(int)).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3).astype(int)).all():
                return True

        # Diagonal checks
        if (
                self.board[0, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[2, 2] == self.player
        ):
            return True
        if (
                self.board[2, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[0, 2] == self.player
        ):
            return True

        # No legal actions means a draw
        if len(self.legal_actions()) == 0:
            return True

        return False

    def render(self):
        print(self.board[::-1])
