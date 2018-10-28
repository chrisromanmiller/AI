import tensorflow as tf
import gym.spaces
import TicTacToe
import numpy as np
import Players

tf.reset_default_graph()


def TicTacToe_model(placeholder, num_actions, scope, reuse=tf.AUTO_REUSE):
    # A model for a TicTacToe q-function
    placeholder = tf.contrib.layers.flatten(placeholder)
    with tf.variable_scope(scope, reuse=reuse):
        out = placeholder
        out = tf.cast(out, tf.float32)
        out = tf.layers.dense(out, 64, bias_initializer=tf.zeros_initializer(), activation=tf.nn.softmax)
        out = tf.layers.dense(out, 64, bias_initializer=tf.zeros_initializer(), activation=tf.nn.softmax)
        out = tf.layers.dense(out, 9, kernel_initializer=tf.zeros_initializer(),
                              bias_initializer=tf.zeros_initializer(), activation=None)
    return out


def peek():
    print("peek")
    test_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model-1")
    for var in test_vars:
        print(var.name, np.max(sess.run(var)))

def policy(observations, model, observation_placeholder):
    dist = tf.distributions.Categorical(logits=model)
    return sess.run(dist.sample(), feed_dict={observation_placeholder: observations})


tf.reset_default_graph()

#define the board, models
observation_placeholder = tf.placeholder(shape = [None, 3,3], dtype = tf.int32)
adv_n_placeholder = tf.placeholder(shape = [None], dtype = tf.float32)
action_placeholder = tf.placeholder(shape = [None], dtype = tf.int32)
new_model = TicTacToe_model(observation_placeholder, 9, scope = "model-1", reuse=tf.AUTO_REUSE)



sess = tf.Session()

sess.run(tf.global_variables_initializer())

peek()

player_NN = Players.NN_Player(new_model, sess, observation_placeholder)

player_NN.peek()

peek()
player_NN.peek()
board = np.array([[1,1,1],[1,1,1],[1,1,0]])
boards = np.array([board, board])

print(str([player_NN.policy(boards) for i in range(2)]))
print(str([policy(boards, new_model, observation_placeholder) for i in range(2)]))
random_Player = Players.Random_Player()

print(str([random_Player.policy(boards) for i in range(2)]))
