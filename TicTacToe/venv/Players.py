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


class NN_Player(Player):
    """Player which uses a NN to dictate policy"""
    def __init__(self, model, model_sample_s, session, observation_placeholder, mask_placeholder, duplicate=True):
        #Keep a fixed model pointer
        self.model = model
        self.model_sample_s = model_sample_s
        #Keep a fixed observation_placehold pointer
        self.observation_placeholder = observation_placeholder
        self.mask_placeholder = mask_placeholder



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


    def policy(self, observations, legal_moves):
        # dist = tf.distributions.Categorical(logits=self.model)
        return self.session.run(self.model_sample_s, feed_dict={self.observation_placeholder: observations, self.mask_placeholder: legal_moves})
