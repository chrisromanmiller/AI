import math
import gym
import itertools
import time
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


#
# """making a observation class wrapper so we can hash for quick look up"""
# class tictactoe_observation():
#
#     def __init__(self, observation):
#         """it is intended the raw_observation be in the form to feed into the neural net"""
#         self.observation = observation
#
#     def __hash__(self):
#         # to something with self.raw_observation
#         hash_value = int(0)
#         row_N = self.observation.shape[0]
#         for (row, col), value in np.ndenumerate(self.observation):
#             hash_value += value << ( 2*(row_N * col + row))
#         return int(hash_value)
#
#
#     def __str__(self):
#         # how to print observation to be pretty
#         return str(self.observation)
#
#
#



class mnk_game():
    """
    Description:
        Playing 2 player Tic Tac Toe
    Source:
        This is developed by Nick Ryder and Chris Miller
    Observation:
        Type: Tuple(Discrete(3),Discrete(3),Discrete(3),Discrete(3),Discrete(3),Discrete(3),Discrete(3),Discrete(3),Discrete(3))
        A tuple which describes the contents of the cell of a 3x3 grid read from left to right, top to bottom.
        0 - empty
        1 - An O marker for player 1
        2 - An X marker for player 2

    Actions:
        Type: Discrete(9)
        Decides which cell of the 3x3 grid to play in, cells are read from left to right, top to bottom.

        Reward is 1 if the game is won, -1 if the game is lost.
    Starting State:
        The 3x3 grid is initalized to have all empty cells, Player 1 goes first.

    Episode Termination:
        Game is Won - 3 O's or 3 X's form a horizontal, vertical, or diagonal line in the 3x3 grid
        Game is Tied - All cells are filled
        Illegal Move - Player tries to play a cell already filled (Counts as win for other player)
    """




    def __init__(self, m = 3, n = 3, k = 3):

        self.m = m
        self.n = n
        self.k = k

        self.action_space = spaces.Discrete(m*n)
        # self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)))


        self.viewer = None
        self.state = None

        self.current_player = None
        self.illegal_moves_N = None
        self.current_winner = None

        self.rewards = None

        self.done = False

    def __str__(self):
        if self.state is None:
            return "Not Initialized"
        else:
            state_strings = np.chararray((self.m,self.n))
            state_strings[:] = '-'
            for i in range(self.m):
                for j in range(self.n):
                    if self.state[0,i,j]:
                        state_strings[i,j] = "0"
                    elif self.state[1,i,j]:
                        state_strings[i,j] = "1"



            return_string = "Current Player: " + str(self.current_player) + "\n"
            return_string += str(state_strings)
        return return_string


    """returns observation of the state for player_id"""
    def get_observation(self):
        if self.current_player == 0:
            return self.state
        if self.current_player == 1:
            return np.logical_not(self.state)

    @staticmethod
    def hash_observation(observation):
        return hash(bytes(obs))
        # hash_value = int(0)
        # row_N = observation.shape[1]
        # for (player_id, row, col), value in np.ndenumerate(observation):
        #     hash_value += int(value) << (2 * (row_N * col + row) + player_id)
        # return int(hash_value)

    """returns the current reward for """
    def get_reward(self):
        return self.rewards[self.current_player]

    """observation and reward should be queried seperately on a per player basis"""
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # state = self.state

        self.rewards[self.current_player] = 0


        # 0 1 2
        # 3 4 5
        # 6 7 8
        action_col = action % self.n
        action_row = int((action - action_col)/self.m)
        #make sure game is not over before attempting to play
        if not self.done:

            assert  (not self.state[0, action_row, action_col] and not self.state[1, action_row, action_col]), "You have played an illegal action. Use legal_moves()"
            self.state[self.current_player, action_row, action_col] = True


        #internal commands which test for termination conditions
        _has_row = self.state_has_row()
        _full_board = np.all(np.logical_or(self.state[0],self.state[1]))

        if _has_row:
            self.done = True
            self.current_winner = self.current_player
            self.rewards[self.current_winner] = 1
        elif _full_board:
            self.done = True
            self.current_winner = 0
            self.rewards = [.5, .5]

        self.current_player = (self.current_player + 1) % 2

        return self.done, {}



    def state_has_row(self):
        """Tests whether the current self.state has a horizontal, vertical, or diagonal row of the same player
            returns boolean"""
        _current_player_state = self.state[self.current_player]
        has_row = False

        #TODO: implement
        # Horizontal Test
        for row in range(self.m):
            if mnk_game._consecutive_and(_current_player_state[row, :]) >= self.k:
                return True
        # Vertical Test
        for col in range(self.n):
            if mnk_game._consecutive_and(_current_player_state[:,col]) >= self.k:
                return True

        for diag in range(self.n):
            _diag_line = np.diagonal(_current_player_state,diag)
            if len(_diag_line) >= self.k and mnk_game._consecutive_and(_diag_line) >= self.k:
                return True

        _lr_current_player_state = np.fliplr(_current_player_state)
        for antidiag in range(self.n):
            _antidiag_line = np.diagonal(_lr_current_player_state,antidiag)
            if len(_antidiag_line) >= self.k and mnk_game._consecutive_and(_antidiag_line) >= self.k:
                return True

        return False

    def legal_moves(self):
        """Returns binary mask on action space for legal moves"""
        return np.logical_not(np.logical_or(self.state[0],self.state[1])).flatten()


    #TODO: Can make faster?
    @staticmethod
    def _consecutive_and(numpy_bool):
        """input is a 1 dim numpy bool array
            determines maximal number of consecutive True"""
        false_index = -1
        max_false_gap = 0
        for index, bool_entry in enumerate(numpy_bool):
            if not bool_entry:
                false_gap = index - false_index - 1
                if false_gap > max_false_gap:
                    max_false_gap = false_gap
                false_index = index

        false_gap = len(numpy_bool) - false_index - 1
        if false_gap > max_false_gap:
            max_false_gap = false_gap
        return max_false_gap

    # @staticmethod
    # def _consecutive_and_test(numpy_bool):
    #     max_true_length = 0
    #     for bit, group_iterator in itertools.groupby(numpy_bool):
    #         group_length = len(list(group_iterator))
    #         if bit and group_length > max_true_length:
    #             max_true_length = group_length
    #     return max_true_length



    def reset(self):
        self.state = np.array([np.zeros((self.m, self.n), dtype=np.bool_),np.zeros((self.m, self.n), dtype=np.bool_)])
        self.current_player = np.random.randint(0,2)
        self.done = False
        self.current_winner = None

        self.rewards = [0,0]


        return self.state



    # def render(self, mode=''):
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = mnk_game()
    env.reset()
    moves = [2,0,4,5,6,7]
    for move in moves:
        env.step(move)
        obs = env.get_observation()
        tic = time.time()
        print(hash(bytes(obs)))
        print(time.time() - tic)

        tic = time.time()
        print(env.hash_observation(obs))
        print(time.time() - tic)





