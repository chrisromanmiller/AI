import tensorflow as tf
import gym.spaces
import TicTacToe
import numpy as np
import Players
import MCTS
from collections import Counter

def batch_rollout(player, opponent, env, max_time_steps=100):
    '''Produces a batch of rollouts from the environment.
    Inputs:
        player: realization of Player.Player abstract class
        opponent: realization of Player.Player abstract class
        env: an environment
        max_time_steps: an integer

    This function plays a number of rounds of a two-player game, and returns the trajectories observed by player

    Returns:
        paths: a list of dictionaries. Each dictionary is a rollout, and takes the keys:
            'observation': [None, obs_dime] np.array of the observations of player
            'action': [None,] np.array of the actions of player
            'reward': [None,] np.array of the rewards gotten by player
        batch_winners: TODO
    '''
    paths = []
    batch_winners = Counter({0: 0, 1: 0, 2: 0})
    time_steps = 0
    while time_steps < max_time_steps:
        path = sample_trajectory(player, opponent, env)
        paths += [path]
        batch_winners[env.current_winner] += 1
        time_steps += len(path['observation'])
    return paths, batch_winners


def sample_trajectory(player, opponent, env):
    """Produces a single rollout of the environment following the player policy
    Inputs:
        player:   realization of Player.Player abstract class
        opponent: realization of Player.Player abstract class
        env:      environment which follows open ai gym environment structure and has a current_player int either 1 or 2
        TODO: it doesn't quite match the reward structure, no?^

    Returns:
    a list of dictionaries. Each dictionary is a rollout, and takes the keys:
        'observation': [None, obs_dime] np.array of the observations of player
        'action': [None,] np.array of the actions of player
        'reward': [None,] np.array of the rewards gotten by player
    """

    obs, acs, rewards, masks = [], [], [], []
    ob = env.reset()
    done = False
    player_has_acted = False
    action = None

    # Do rest of moves
    while not done:
        # Get current observation of current player
        ob = env.get_observation(env.current_player)
        legal_moves = env.legal_moves()
        if env.current_player == 1:
            # Reward is recorded as results of state,action pair... need to check player 1 has acted already
            if player_has_acted:
                rewards.append(env.get_reward(1))
            else:
                player_has_acted = True

            action = player.policy(np.array([ob]), np.array([legal_moves]))
            obs.append(ob)
            acs.append(action[0])
            masks.append(legal_moves)
        else:
            action = opponent.policy(np.array([ob]), np.array([legal_moves]))
        done, _ = env.step(action[0])

        # Need to record final reward for player 1
    rewards.append(env.get_reward(1))

    path = {"observation": np.array(obs, dtype=np.int32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.int32),
            "mask": np.array(masks, dtype=np.int32)}
    return path


tf.reset_default_graph()

env = TicTacToe.TicTacToe()
env.reset()

mcts = MCTS.mcts()

child = Players.Child_Player()
mcts_player = Players.MCTS_Player(mcts)
for x in range(1000):
    if x % 100 == 0:
        _, stats = batch_rollout(child, mcts_player, env)
        print(stats)
    mcts.rollout(env)



# def TicTacToe_model(placeholder, num_actions, scope, reuse=tf.AUTO_REUSE):
#     # A model for a TicTacToe q-function
#     placeholder = tf.contrib.layers.flatten(placeholder)
#     with tf.variable_scope(scope, reuse=reuse):
#         out = placeholder
#         out = tf.cast(out, tf.float32)
#         out = tf.layers.dense(out, 64, bias_initializer=tf.zeros_initializer(), activation=tf.nn.softmax)
#         out = tf.layers.dense(out, 64, bias_initializer=tf.zeros_initializer(), activation=tf.nn.softmax)
#         out = tf.layers.dense(out, 9, kernel_initializer=tf.zeros_initializer(),
#                               bias_initializer=tf.zeros_initializer(), activation=None)
#     return out
#
#
# def peek():
#     print("peek")
#     test_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model-1")
#     for var in test_vars:
#         print(var.name, np.max(sess.run(var)))
#
# def policy(observations, model, observation_placeholder):
#     dist = tf.distributions.Categorical(logits=model)
#     return sess.run(dist.sample(), feed_dict={observation_placeholder: observations})
#
#
# tf.reset_default_graph()
#
# #define the board, models
# observation_placeholder = tf.placeholder(shape = [None, 3,3], dtype = tf.int32)
# adv_n_placeholder = tf.placeholder(shape = [None], dtype = tf.float32)
# action_placeholder = tf.placeholder(shape = [None], dtype = tf.int32)
# new_model = TicTacToe_model(observation_placeholder, 9, scope = "model-1", reuse=tf.AUTO_REUSE)
#
#
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# peek()
#
# player_NN = Players.NN_Player(new_model, sess, observation_placeholder)
#
# player_NN.peek()
#
# peek()
# player_NN.peek()
# board = np.array([[1,1,1],[1,1,1],[1,1,0]])
# boards = np.array([board, board])
#
# print(str([player_NN.policy(boards) for i in range(2)]))
# print(str([policy(boards, new_model, observation_placeholder) for i in range(2)]))
# random_Player = Players.Random_Player()
#
# print(str([random_Player.policy(boards) for i in range(2)]))
