from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class Player(ABC):

    def __init__(self):
        super().__init__()

        @abstractmethod
        def policy(self, observations, legal_moves):
            """ Input:  [None, obs_dim] array of observations, first axis has size batch_N
                Output: [None, ac_dim] ndarray of actions, first axis has size batch_N """
            pass


class Human_Player(Player):
    """A player class which shows an input and waits for legal move"""


    def policy(self, observations, legal_moves):
        actions = []
        for observation, legal_move in zip(observations, legal_moves):
            is_legal_move = False
            print(observation)
            while not is_legal_move:
                action = int(input("Input a legal move:"))
                is_legal_move = legal_move[action]
            actions.append(action)
        return actions


class Random_Player(Player):
    """A player class which randomly picks a legal move"""


    def policy(self, observations, legal_moves):
        """Currently poorly coded: doesnt use legal moves"""

        import numpy as np
        actions = []
        for observation, legal_move, in zip(observations, legal_moves):
            legal_move_indices = np.where(legal_move)[0]

            random_action = None
            if len(legal_move_indices) > 0:
                random_action = np.random.choice(legal_move_indices)
            actions.append(random_action)


        return actions


class Expert_TicTacToe_Player(Player):
    """A player class that plays a single deterministic policy mimicking this xkcd comic: https://xkcd.com/832/"""

    def policy(self, observations, legal_moves):
        """	0 1 2
            3 4 5
            6 7 8 """
        actions = []
        for observation, legal_moves in zip(observations, legal_moves):
            action = None
            observation_one_two_rep = observation[0].astype(int) + 2*observation[1].astype(int)
            observation_flatten_list = observation_one_two_rep.flatten().tolist()

            player_moves_N = observation_flatten_list.count(1)
            opponent_moves_N = observation_flatten_list.count(2)
            # handle edge cases
            # check for blocking moves
            # check for winning moves
            # last action to be set is the action taken ^reverse order of importance

            if player_moves_N == 0 and opponent_moves_N == 0:
                action = 0
            if player_moves_N == 0 and opponent_moves_N == 1:
                if legal_moves[4] == 1:
                    action = 4
                else:
                    action = 0

            if player_moves_N == 1 and opponent_moves_N == 1:
                if observation_flatten_list[6] == 2:
                    action = 2
                elif observation_flatten_list[2] == 2:
                    action = 6
                elif observation_flatten_list[4] == 2:
                    action = 8
                elif observation_flatten_list[8] == 2:
                    action = 2
                else:
                    action = 4

            if opponent_moves_N == 2 and player_moves_N == 1:

                """X
                    O
                     X"""
                if observation_flatten_list[0] == 2 and observation_flatten_list[8] == 2 and observation_flatten_list[
                    4] == 1:
                    action = 7
                if observation_flatten_list[2] == 2 and observation_flatten_list[6] == 2 and observation_flatten_list[
                    4] == 1:
                    action = 7
                """O
                    X
                     O"""
                if observation_flatten_list[0] == 2 and observation_flatten_list[4] == 2 and observation_flatten_list[
                    8] == 1:
                    action = 6
                if observation_flatten_list[0] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[
                    8] == 2:
                    action = 6
                if observation_flatten_list[2] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[
                    6] == 2:
                    action = 0
                if observation_flatten_list[6] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[
                    2] == 2:
                    action = 0
                """X
                    O
                    X"""
                if observation_flatten_list[4] == 1:
                    if observation_flatten_list[0] == 2 and observation_flatten_list[8] == 0:
                        action = 8
                    if observation_flatten_list[8] == 2 and observation_flatten_list[0] == 0:
                        action = 0
                    if observation_flatten_list[6] == 2 and observation_flatten_list[2] == 0:
                        action = 2
                    if observation_flatten_list[2] == 2 and observation_flatten_list[6] == 0:
                        action = 6
                """ X
                    O
                    X"""
                if observation_flatten_list[4] == 1 and observation_flatten_list[1] == 2 and observation_flatten_list[
                    7] == 2:
                    action = 5
                if observation_flatten_list[4] == 1 and observation_flatten_list[3] == 2 and observation_flatten_list[
                    5] == 2:
                    action = 1
                """ X
                   XO"""
                if observation_flatten_list[4] == 1 and observation_flatten_list[1] == 2:
                    if observation_flatten_list[5] == 2:
                        action = 2
                    if observation_flatten_list[3] == 2:
                        action = 0
                if observation_flatten_list[4] == 1 and observation_flatten_list[7] == 2:
                    if observation_flatten_list[5] == 2:
                        action = 8
                    if observation_flatten_list[3] == 2:
                        action = 6

            if opponent_moves_N == 2 and player_moves_N == 2:
                if observation_flatten_list[4] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[
                    1] == 2:
                    action = 6
                elif observation_flatten_list[4] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[
                    3] == 2:
                    action = 2
                if observation_flatten_list[1] == 2 and observation_flatten_list[2] == 1:
                    action = 8
                if observation_flatten_list[3] == 2 and observation_flatten_list[6] == 1:
                    action = 8
                if observation_flatten_list[1] == 2 and observation_flatten_list[2] == 1 and observation_flatten_list[
                    8] == 2:
                    action = 6

            if opponent_moves_N == 3 and player_moves_N == 2:
                """o  
                   xxo
                     x"""
                if observation_flatten_list[1] == 0:
                    if observation_flatten_list[0] == 1 and observation_flatten_list[2] == 0:
                        action = 2
                    if observation_flatten_list[2] == 1 and observation_flatten_list[0] == 0:
                        action = 0
                if observation_flatten_list[3] == 0:
                    if observation_flatten_list[0] == 1 and observation_flatten_list[6] == 0:
                        action = 6
                    if observation_flatten_list[6] == 1 and observation_flatten_list[0] == 0:
                        action = 0
                if observation_flatten_list[5] == 0:
                    if observation_flatten_list[2] == 1 and observation_flatten_list[8] == 0:
                        action = 8
                    if observation_flatten_list[2] == 0 and observation_flatten_list[8] == 1:
                        action = 2
                if observation_flatten_list[7] == 0:
                    if observation_flatten_list[6] == 1 and observation_flatten_list[8] == 0:
                        action = 8
                    if observation_flatten_list[6] == 0 and observation_flatten_list[8] == 1:
                        action = 6
                """ x  
                   xoo
                    x """
                if observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2 and observation_flatten_list[
                    7] == 2 and observation_flatten_list[4] == 1 and observation_flatten_list[5] == 1:
                    action = 2
                if observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2 and observation_flatten_list[
                    4] == 1 and observation_flatten_list[5] == 2 and observation_flatten_list[7] == 1:
                    action = 6
                if observation_flatten_list[1] == 2 and observation_flatten_list[5] == 2 and observation_flatten_list[
                    7] == 2 and observation_flatten_list[4] == 1 and observation_flatten_list[3] == 1:
                    action = 0
                if observation_flatten_list[3] == 2 and observation_flatten_list[7] == 2 and observation_flatten_list[
                    5] == 2 and observation_flatten_list[1] == 1 and observation_flatten_list[4] == 1:
                    action = 0

                """x   
                    ox
                    xo"""
                if observation_flatten_list[4] == 1 and observation_flatten_list[8] == 1 and observation_flatten_list[
                    0] == 2 and observation_flatten_list[5] == 2 and observation_flatten_list[7] == 2:
                    action = 2
                if observation_flatten_list[4] == 1 and observation_flatten_list[0] == 1 and observation_flatten_list[
                    8] == 2 and observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2:
                    action = 2
                if observation_flatten_list[4] == 1 and observation_flatten_list[2] == 1 and observation_flatten_list[
                    6] == 2 and observation_flatten_list[1] == 2 and observation_flatten_list[5] == 2:
                    action = 0
                if observation_flatten_list[4] == 1 and observation_flatten_list[2] == 2 and observation_flatten_list[
                    6] == 1 and observation_flatten_list[3] == 2 and observation_flatten_list[7] == 2:
                    action = 0

                """xox
                    o
                    x"""
                if observation_flatten_list[4] == 1 and observation_flatten_list[0] == 2 and observation_flatten_list[
                    1] == 1 and observation_flatten_list[2] == 2 and observation_flatten_list[7] == 2:
                    action = 3
                if observation_flatten_list[4] == 1 and observation_flatten_list[0] == 2 and observation_flatten_list[
                    3] == 1 and observation_flatten_list[6] == 2 and observation_flatten_list[5] == 2:
                    action = 1
                if observation_flatten_list[4] == 1 and observation_flatten_list[2] == 2 and observation_flatten_list[
                    5] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[3] == 2:
                    action = 1
                if observation_flatten_list[4] == 1 and observation_flatten_list[6] == 2 and observation_flatten_list[
                    7] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[1] == 2:
                    action = 3

            if player_moves_N == 3 and opponent_moves_N == 4:
                action = np.where(legal_moves)[0][0]

            lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

            # block moves
            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(2) == 2:
                    # find winning move:
                    action = line[board_line_values.index(0)]
                # print("found block")

            # win game!
            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(1) == 2:
                    # find winning move:
                    action = line[board_line_values.index(0)]
                # print("found winning move")

            assert action is not None, print("No action for Expert Policy", observation)
            actions.append(action)
        return actions




class Child_Player(Player):
    """A player class that blocks any two consecutive squares. Otherwise, it plays randomly"""

    def policy(self, observations, legal_moves):
        # block moves

        actions = []
        for observation, legal_move, in zip(observations, legal_moves):
            observation_one_two_rep = observation[0].astype(int) + 2*observation[1].astype(int)
            observation_flatten_list = observation_one_two_rep.flatten().tolist()
            legal_move_indices = np.where(legal_move)[0]
            action = np.random.choice(legal_move_indices)


            lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(2) == 2:
                    # find winning move:
                    action = line[board_line_values.index(0)]
                # print("found block")

            # win game!
            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(1) == 2:
                    # find winning move:
                    action = line[board_line_values.index(0)]
                # print("found winning move")
            actions.append(action)

        return actions




class NN_Player(Player):
    """Player which uses a NN to dictate policy
        model_sample_s symbolic operation to get probability distribution for actions
        determistic determines whether to arg max or draw"""

    def __init__(self, model, model_sample_s, session, observation_placeholder, duplicate=True, deterministic=True):
        # Keep a fixed model pointer
        self.model = model
        self.model_sample_s = model_sample_s
        # Keep a fixed observation_placehold pointer
        self.observation_placeholder = observation_placeholder
        self.deterministic = deterministic

        if duplicate:
            print("duplicating session to freeze weights for evaluation...")
            temp_file_name = './to_duplicate.ckpt'

            # Want to duplicate session
            saver = tf.train.Saver()
            saver.save(session, temp_file_name)
            self.session = tf.Session()
            saver.restore(self.session, temp_file_name)
        else:
            print("Warning: not duplicating session, evaluation will change with tf updates")
            self.session = session

    def __del__(self):
        """Need to destroy tensorflow session"""
        print("Destroying NN_Player and Session...")
        self.session.close()

    def peek(self):
        print("NN_Player Peek")
        test_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model-1")
        for var in test_vars:
            print(var.name, np.max(self.session.run(var)))

    def policy(self, observations, legal_moves, epsilon=0):
        """evaluates model_sample_s with probability 1 - eps
            returns random legal_move with probability eps"""
        distributions = self.session.run(self.model_sample_s,
                                         feed_dict={self.observation_placeholder: observations})
        actions = []
        for legal_move, dist in zip(legal_moves, distributions):
            indices = np.where(legal_move)[0]
            if self.deterministic:
                random_float = np.random.rand()
                if random_float >= epsilon:
                    values = dist[indices]
                    arg_max = np.argmax(values)
                    actions.append(indices[arg_max])
                else:
                    actions.append(np.random.choice(indices))
            else:
                #TODO: TEST
                legal_probabilities = np.multiply(dist, legal_move)
                action = np.random.choice(range(len(legal_probabilities)),legal_probabilities)
                actions.append(action)

        return actions



class MCTS_Player(Player):
    """Player class which uses a Monte Carlo Tree Search algorithm to select actions"""


    def __init__(self, mcts):
        # Keep a fixed model pointer
        self.mcts = mcts



    def policy(self, observations, legal_moves):

        import MCTS
        import tictactoe
        """ uses the exploitation part of MCTS to evaluate
            if state not seen before it picks a random action"""
        actions = []
        for observation, legal_move in zip(observations, legal_moves):
            observation_hash = tictactoe.mnk_game.hash_observation(observation)
            try:
                mcts_node = self.mcts.states[observation_hash]
            except KeyError:
                mcts_node = MCTS.mcts_node_(observation, legal_move)
            actions.append(mcts_node.exploit_policy())
        return actions
