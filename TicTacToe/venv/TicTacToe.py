import math
import gym
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



class TicTacToe():
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
            state_strings = np.chararray((3,3))
            for i in range(3):
                for j in range(3):
                    if self.state[i,j] == 0:
                        state_strings[i,j] = ""
                    else:
                        state_strings[i,j] = str(self.state[i, j])



            return_string = "Current Player: " + str(self.current_player) + "\n"
            # return_string += state_strings[0,0] + "|" + state_strings[0,1] + "|" + state_strings[0,2] +"\n"
            # return_string += "-----\n"
            # return_string += state_strings[1,0] + "|" + state_strings[1,1] + "|" + state_strings[1,2] + "\n"
            # return_string += "-----\n"
            # return_string += state_strings[2,0] + "|" + state_strings[2,1] + "|" + state_strings[2,2] + "\n"
            return_string += str(state_strings)
        return return_string
    # def seed(self):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    """returns observation of the state for player_id"""
    def get_observation(self, player_id):
        current_observation = np.array(self.state)
        if player_id == 2:
            current_observation = self.flip_state(current_observation)
        # if hashable:
        #     return tictactoe_observation(current_observation)
        # else:
        return current_observation

    @staticmethod
    def hash_observation(observation):
        hash_value = int(0)
        row_N = observation.shape[0]
        for (row, col), value in np.ndenumerate(observation):
            hash_value += value << (2 * (row_N * col + row))
        return int(hash_value)

    """returns the current reward for """
    def get_reward(self, player_id):
        return self.rewards[player_id-1]

    """observation and reward should be queried seperately on a per player basis"""
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # state = self.state

        self.rewards[self.current_player - 1] = 0


        # 0 1 2
        # 3 4 5
        # 6 7 8
        action_col = action % 3
        action_row = int((action - action_col)/3)

        #make sure game is not over before attempting to play
        if not self.done:
            assert self.state[action_row, action_col] == 0, print("You have played an illegal action. Use legal_moves()", self.state,self.get_observation(self.current_player), action)
#             if self.state[action_row, action_col] != 0:
#
#                 # The action is on a cell already occupied, illegal move negative reward
#                 self.illegal_moves_N[self.current_player-1] += 1
#                 reward += -1
# #                self.done = True
#             else:
            self.state[action_row, action_col] = self.current_player


        #test for full board
        if np.min(self.state) > 0:
            self.done = True
            self.current_winner = 0
            self.rewards[self.current_winner - 1] = .5
            self.rewards[self.current_winner % 2] = .5

        #internal command which tests for win condition
        winner_id = self.state_has_row()
        if winner_id != 0:
            self.done = True
            self.current_winner = winner_id
            self.rewards[self.current_winner - 1] = 1
            self.rewards[self.current_winner % 2] = 0
            # if self.current_player == winner_id:
            #     reward += 5
            # else:
            #     reward += -1



        #swap players and flip state for player 2
        # current_observation = np.array(self.state)
        # if self.current_player == 1:
        #     self.current_player = 2
        #     current_observation = self.flip_state(current_observation)
        # else:
        #     self.current_player = 1
        self.current_player = (self.current_player % 2) + 1

        return self.done, {}


    def flip_state(self, alter_state):
        for row in range(3):
            for col in range(3):
                if alter_state[row, col] == 1:
                    alter_state[row, col] = 2
                elif alter_state[row, col] == 2:
                    alter_state[row, col] = 1
        return alter_state


    def state_has_row(self):
        """Tests whether the current self.state has a horizontal, vertical, or diagonal row of the same player
            returns boolean"""
        row_player_id = 0
        current_player_state = self.state[self.current_player]


        has_row = False

        #TODO: implement
        # Horizontal Test
        for row in range(self.m):
            if np.all(current_player_state[row, :]):
                has_row = True
        # Vertical Test
        for col in range(self.n):
            if np.all(current_player_state[:, col]):
                has_row = True

        #Diagonal Tests
        if self.state[0,0] == self.state[1,1] and self.state[1,1] == self.state[2,2] and self.state[0,0] != 0:
            row_player_id = self.state[0,0]
        if self.state[0,2] == self.state[1,1] and self.state[1,1] == self.state[2,0] and self.state[0,2] != 0:
            row_player_id = self.state[0,2]
        return int(row_player_id)

    def legal_moves(self):
        """Returns binary mask on action space for legal moves"""
        state_flatten =  self.state.flatten()
        for index in range(len(state_flatten)):
            if state_flatten[index] != 0:
                state_flatten[index] = 0
            else:
                state_flatten[index] = 1
        return np.array(state_flatten)

    def _k_consecutive_and(self, numpy_bool):
        """input is a 1 dim numpy bool array
            determines whether k consecutive entries are True"""
        


    def reset(self):
        self.state = [np.zeros((self.m, self.n), dtype=np.bool_),np.zeros((self.m, self.n), dtype=np.bool_)]
        self.current_player = np.random.randint(0,2)
        self.done = False
        self.current_winner = None

        self.rewards = [0,0]


        return np.array(self.state)



    def render(self, mode=''):
        # screen_width = 600
        # screen_height = 400
        #
        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        #
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        #     l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        #     axleoffset = cartheight / 4.0
        #     cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     self.carttrans = rendering.Transform()
        #     cart.add_attr(self.carttrans)
        #     self.viewer.add_geom(cart)
        #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        #     pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        #     pole.set_color(.8, .6, .4)
        #     self.poletrans = rendering.Transform(translation=(0, axleoffset))
        #     pole.add_attr(self.poletrans)
        #     pole.add_attr(self.carttrans)
        #     self.viewer.add_geom(pole)
        #     self.axle = rendering.make_circle(polewidth / 2)
        #     self.axle.add_attr(self.poletrans)
        #     self.axle.add_attr(self.carttrans)
        #     self.axle.set_color(.5, .5, .8)
        #     self.viewer.add_geom(self.axle)
        #     self.track = rendering.Line((0, carty), (screen_width, carty))
        #     self.track.set_color(0, 0, 0)
        #     self.viewer.add_geom(self.track)
        #
        #     self._pole_geom = pole
        #
        # if self.state is None: return None
        #
        # # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])
        #
        #
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == 'main':
    env = TicTacToe()
    env.reset()
    print(env)

    while True:
        action = int(input(" 0 1 2 \n 3 4 5 \n 6 7 8"))
        current_observation, reward, done, what = env.step(action)
        if done:
            print("DONEDONEDONE")
        print(env)
