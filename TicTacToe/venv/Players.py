from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class Player(ABC):

    def __init__(self):
        super().__init__()

        @abstractmethod
        def policy(self, observations, legal_moves):
            """ Input:  ndarray of observations, first axis has size batch_N
                Output: ndarray of actions, first axis has size batch_N """
            pass



class Human_Player(Player):


    def policy(self, observations, legal_moves):
        #Get batchsize
        observations_N = observations.shape[0]
        print(observations_N)
        actions = np.ndarray((observations_N,), int)
        for index in range(observations_N):
            legal_move = False
            print(observations[index])
            # while not legal_move:
            #     try: input = raw_input
            #     except NameError: pass
            in_ = input("Say something: ")

            actions[index] = in_
            legal_move = (legal_moves[index][actions[index]] == 1)
        print(actions)
        return actions

class Random_Player(Player):
    def policy(self, observations, legal_moves):
        """Currently poorly coded: doesnt use legal moves"""

        import numpy as np
        #Get batchsize
        observations_N = observations.shape[0]

        #Flattens observations
        observation_flatten = np.reshape(observations, (observations_N, -1))
        #Picks out nonzero entry indices
        observation_flatten_nz = [[index for index in range(len(observation_flatten[obs_ind])) if observation_flatten[obs_ind][index] == 0] for obs_ind in range(observations_N)]
        actions = np.ndarray((observations_N,),int)
        for obs_ind in range(observations_N):
            #Returns a random nonzero entry index
            if len(observation_flatten_nz[obs_ind]) > 0:
                actions[obs_ind] = observation_flatten_nz[obs_ind][np.random.randint(0,len(observation_flatten_nz[obs_ind]))]
            else:
                actions[obs_ind] = 0
        return actions

class Expert_Player(Player):


    def policy(self, observations, legal_moves):
        """	0 1 2
            3 4 5
            6 7 8 """
        actions = []
        for observation, legal_moves in zip(observations, legal_moves):
            action = None
            observation_flatten_list = observation.flatten().tolist()
            player_moves_N = observation_flatten_list.count(1)
            opponent_moves_N = observation_flatten_list.count(2)
            #handle edge cases
            #check for blocking moves
            #check for winning moves
            #last action to be set is the action taken ^reverse order of importance


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
                if observation_flatten_list[0] == 2 and observation_flatten_list[8] == 2 and observation_flatten_list[4] == 1:
                    action = 7
                if observation_flatten_list[2] == 2 and observation_flatten_list[6] == 2 and observation_flatten_list[4] == 1:
                    action = 7
                """O
                    X
                     O"""
                if observation_flatten_list[0] == 2 and observation_flatten_list[4] == 2 and observation_flatten_list[8] == 1:
                    action = 6
                if observation_flatten_list[0] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[8] == 2:
                    action = 6
                if observation_flatten_list[2] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[6] == 2:
                    action = 0
                if observation_flatten_list[6] == 1 and observation_flatten_list[4] == 2 and observation_flatten_list[2] == 2:
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
                if observation_flatten_list[4] == 1 and observation_flatten_list[1] == 2 and observation_flatten_list[7] == 2:
                    action = 5
                if observation_flatten_list[4] == 1 and observation_flatten_list[3] == 2 and observation_flatten_list[5] == 2:
                    action = 1
                """ X
                   XO"""
                if observation_flatten_list[4] ==1 and observation_flatten_list[1] == 2:
                    if observation_flatten_list[5] == 2:
                        action = 2
                    if observation_flatten_list[3] ==2:
                        action = 0
                if observation_flatten_list[4] ==1 and observation_flatten_list[7] == 2:
                    if observation_flatten_list[5] == 2:
                        action = 8
                    if observation_flatten_list[3] ==2:
                        action = 6





            if opponent_moves_N == 2 and player_moves_N == 2:
                if observation_flatten_list[4] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[1] == 2:
                    action = 6
                elif observation_flatten_list[4] == 1 and observation_flatten_list[8] ==2 and observation_flatten_list[3] ==2:
                    action = 2
                if observation_flatten_list[1] == 2 and observation_flatten_list[2] == 1:
                    action = 8
                if observation_flatten_list[3] == 2 and observation_flatten_list[6] == 1:
                    action = 8
                if observation_flatten_list[1] == 2 and observation_flatten_list[2] == 1 and observation_flatten_list[8] == 2:
                    action = 6

            if opponent_moves_N == 3 and player_moves_N == 2:
                """o  
                   xxo
                     x"""
                if observation_flatten_list[1] == 0:
                    if observation_flatten_list[0] == 1 and observation_flatten_list[2] == 0:
                        action = 2
                    if observation_flatten_list[2]  ==1 and observation_flatten_list[0] == 0:
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
                if observation_flatten_list[7]== 0:
                    if observation_flatten_list[6] == 1 and observation_flatten_list[8] == 0:
                        action = 8
                    if observation_flatten_list[6] == 0 and observation_flatten_list[8] == 1:
                        action = 6
                """ x  
                   xoo
                    x """
                if observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2 and observation_flatten_list[7] == 2 and observation_flatten_list[4] == 1 and observation_flatten_list[5] == 1:
                    action = 2
                if observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2 and observation_flatten_list[4] == 1 and observation_flatten_list[5] == 2 and observation_flatten_list[7] == 1:
                    action = 6
                if observation_flatten_list[1] == 2 and observation_flatten_list[5] == 2 and observation_flatten_list[7] == 2 and observation_flatten_list[4] == 1 and observation_flatten_list[3] == 1:
                    action = 0
                if observation_flatten_list[3] == 2 and observation_flatten_list[7] == 2 and observation_flatten_list[5] == 2 and observation_flatten_list[1] == 1 and observation_flatten_list[4] == 1:
                    action = 0

                """x   
                    ox
                    xo"""
                if observation_flatten_list[4] == 1 and observation_flatten_list[8] == 1 and observation_flatten_list[0] == 2 and observation_flatten_list[5] == 2 and observation_flatten_list[7] == 2:
                    action = 2
                if observation_flatten_list[4] == 1 and observation_flatten_list[0] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[1] == 2 and observation_flatten_list[3] == 2:
                    action = 2
                if observation_flatten_list[4] == 1 and observation_flatten_list[2] == 1 and observation_flatten_list[6] == 2 and observation_flatten_list[1] == 2 and observation_flatten_list[5] == 2:
                    action = 0
                if observation_flatten_list[4] == 1 and observation_flatten_list[2] == 2 and observation_flatten_list[6] == 1 and observation_flatten_list[3] == 2 and observation_flatten_list[7] == 2:
                    action = 0

                """xox
                    o
                    x"""
                if observation_flatten_list[4] ==1 and observation_flatten_list[0] == 2 and observation_flatten_list[1] == 1 and observation_flatten_list[2] == 2 and observation_flatten_list[7] == 2:
                    action = 3
                if observation_flatten_list[4] ==1 and observation_flatten_list[0] == 2 and observation_flatten_list[3] == 1 and observation_flatten_list[6] == 2 and observation_flatten_list[5] == 2:
                    action = 1
                if observation_flatten_list[4] ==1 and observation_flatten_list[2] == 2 and observation_flatten_list[5] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[3] == 2:
                    action = 1
                if observation_flatten_list[4] ==1 and observation_flatten_list[6] == 2 and observation_flatten_list[7] == 1 and observation_flatten_list[8] == 2 and observation_flatten_list[1] == 2:
                    action = 3

            if player_moves_N == 3 and opponent_moves_N == 4:
                action = np.where(legal_moves)[0][0]



            lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]


            #block moves
            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(2) == 2:
                    #find winning move:
                    action = line[board_line_values.index(0)]
                # print("found block")

            #win game!
            for line in lines:
                board_line_values = []
                for entry in line:
                    board_line_values.append(observation_flatten_list[entry])
                if board_line_values.count(0) > 0 and board_line_values.count(1) == 2:
                    #find winning move:
                    action = line[board_line_values.index(0)]
                # print("found winning move")

            assert action is not None, print("No action for Expert Policy", observation)
            actions.append(action)
        return actions




class NN_Player(Player):
    """Player which uses a NN to dictate policy
        model_sample_s symbolic operation to get probability distribution for actions
        determistic determines whether to arg max or draw"""
    def __init__(self, model, model_sample_s, session, observation_placeholder, duplicate=True, deterministic = True):
        #Keep a fixed model pointer
        self.model = model
        self.model_sample_s = model_sample_s
        #Keep a fixed observation_placehold pointer
        self.observation_placeholder = observation_placeholder
        self.deterministic = deterministic



        if duplicate:
            print("duplicating session to freeze weights for evaluation...")
            temp_file_name = './to_duplicate.ckpt'


            #Want to duplicate session
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


    def policy(self, observations, legal_moves, epsilon = 0):
        """evaluates model_sample_s with probability 1 - eps
            returns random legal_move with probability eps"""
        random_float = np.random.rand()
        if random_float >= epsilon:
            distributions = self.session.run(self.model_sample_s,
                                    feed_dict={self.observation_placeholder: observations})
            if self.deterministic:
                actions = []
                for legal_move, dist in zip(legal_moves, distributions):
                    indices = np.where(legal_move)[0]
                    values = dist[indices]
                    arg_max = np.argmax(values)
                    actions.append(indices[arg_max])
                return actions
            else:
                return
        else:
            return np.random.choice(np.where(legal_moves))



#
#
# class Q_Player(Player):
#     """Player which uses Q-learning to dictate policy"""
#     def __init__(self, model, session, observation_placeholder, mask_placeholder, duplicate=True):
#         #Keep a fixed model pointer
#         self.model = model
#
#         #Keep a fixed observation_placehold pointer
#         self.observation_placeholder = observation_placeholder
#         self.mask_placeholder = mask_placeholder
#
#         #Q-learning requires a method to select the best legal move
#         indices = tf.where(self.mask_placeholder)
#         values = tf.gather_nd(model, indices)
#         denseShape = tf.cast(tf.shape(model), tf.int64)
#         x = tf.SparseTensor(indices, values, denseShape)
#         x = tf.scatter_nd(x.indices, x.values, x.dense_shape)
#         self.prediction =  tf.argmax(x, 1)
#
#
#         if duplicate:
#             print("duplicating session to freeze weights for evaluation...")
#             temp_file_name = './to_duplicate.ckpt'
#
#
#             #Want to duplicate session
#             saver = tf.train.Saver()
#             saver.save(session, temp_file_name)
#             self.session = tf.Session()
#             saver.restore(self.session, temp_file_name)
#         else:
#             print("Warning: not duplicating session, evaluation will change with tf updates")
#             self.session = session
#
#     def __del__(self):
#         """Need to destroy tensorflow session"""
#         print("Destroying Q_Player and Session...")
#         self.session.close()
#
#     def peek(self):
#         print("NN_Player Peek")
#         test_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model-1")
#         for var in test_vars:
#             print(var.name, np.max(self.session.run(var)))
#
#
#     def policy(self, observations, legal_moves):
#         return self.session.run(self.prediction, feed_dict={self.observation_placeholder: observations, self.mask_placeholder: legal_moves})
#
#
#     def q_function(self, observations):
#         return self.session.run(self.model, feed_dict={self.observation_placeholder: observations})
#
