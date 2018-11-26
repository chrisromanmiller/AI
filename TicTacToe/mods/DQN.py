#A module that contains the functions that define a Q-network

import tensorflow as tf
import numpy as np


def TicTacToe_model(observation_placeholder, scope, num_actions = 9):
    '''A model for a TicTacToe Q-function
    Inputs:
        observation_placeholder: [None, ob_dim] placeholder representing inputs to our neural network
        scope: a string that becomes the scope of all layers in this network
        reuse: 
        num_actions: an int representing the number of possible actions (the output dimension)
    
    The final layer outputs values in the range [-1,1], which matches the range of possible target q-values
    placeholder = tf.contrib.layers.flatten(placeholder)
    
    The Q-function is thought of as a function of two varables Q(s,a). Here we treat it as a num_actions-dimensional
    function of one variable, so that Q(s,a) = Q(s)[a]
    
    We initialize bias and weights to zero, except for the final layer, where the weights are initialized to one.  
    
    Returns:
        model: [None, num_actions] variable representing the outputs of our q-function
    '''
    with tf.variable_scope(scope):
        out = tf.contrib.layers.flatten(observation_placeholder)
        out = tf.contrib.layers.flatten(observation_placeholder)        
        out = tf.cast(out, tf.float32)
        out = tf.layers.dense(out, 64  , bias_initializer = tf.zeros_initializer(), activation = tf.nn.softmax)
        out = tf.layers.dense(out, 64  , bias_initializer = tf.zeros_initializer(), activation = tf.nn.softmax)
        out = tf.layers.dense(out, 64  , bias_initializer = tf.zeros_initializer(), activation = tf.nn.softmax)
        out = tf.layers.dense(out, num_actions , kernel_initializer = tf.zeros_initializer(), bias_initializer = tf.zeros_initializer(), activation = tf.nn.sigmoid)
        out = (out*2)-1
    return out


def symbolic_Q_update(model, target_placeholder, action_placeholder, learning_rate = .01):
    '''Produce the symbolic variables for loss, the update, and the optimizer
    Inputs:
        model: [None, action_dim] variable consisting of Q values
        target_placeholder: [None,] placeholder that will be fed target values for the Q-function
        action_placeholder: [None,] placeholder that will be fed the action values
        learning_rate: float representing the size of gradient step
        
        The loss is the mean squared error ||Q(s,a) - Q'(s,a)||^2
        We use AdamOptimizer with no bells or whistles
        
    Returns:
        update_op: a method to be called when we desire to take a gradient step
    '''
    q_action_s = tf.reduce_sum(tf.multiply(model, tf.one_hot(action_placeholder, 9)), 1)
    loss = tf.losses.mean_squared_error(q_action_s, target_placeholder)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_op = optimizer.minimize(loss)
    return update_op, loss

def compute_target_values(player, next_state, masks, not_end_of_path, reward, decay = .99, verbose = False):
    '''Computes the target values for our Q-function update
    Inputs: 
        model: [None, action_dim] variable consisting of Q values
        next_state: [None, ob_dim] np.array of states
        masks: [None, ac_dim] np.array of masks (legal moves)
        not_end_of_path: [None,] np.array of 0,1 integers (0 denotes the end of a rolllout)
        reward: [None,] np.array of real numbers representing the return of the action resulting in next_state
        decay: real number in [0,1] representing the decay rate (often called gamma)
        verbose: Boolean
    
    This function is used to compute the Bellman backup values used to update our Q-function. 
        Recall: Q(s,a,s') <~~  r(s,a) +  max_a' [Q(s',a')]
        The right side of this equation is called the target value.
    
    Returns:
        target: [None,] batch of real numbers, indicating target values
    ''' 
    next_state_Qs = player.session.run(player.model, feed_dict= {player.observation_placeholder: next_state})
    future_expected_reward = []
    for next_state_Q, mask in zip(next_state_Qs,masks):
        indices = np.where(mask)
        values = next_state_Q[indices]
        future_expected_reward.append(np.max(values))
    future_reward_if_not_done = [eop * fer for eop, fer in zip(not_end_of_path.tolist(), future_expected_reward)]
    target = reward + future_reward_if_not_done
    if verbose:
        print("not end of path", not_end_of_path)
        print("future expected reward", future_expected_reward)
        print("future reward if not done", future_reward_if_not_done)
        print("reward", reward)
        print("target", target)
        print("--")
    return target

def sample_paths(paths, batch_size = 10):
    '''From a collection of rollouts, this samples a random uniform batch
    Inputs: 
        paths: a list of dictionaries containing the data of a rollout
        batch_size: integer determining batch size to be returned
    
    Returns:
        state1: [batch_size, ob_dim] np.array of states
        action: [batch_size,] np.array of actions
        state2: [batch_size, ob_dim] np.array of states
        reward: [batch_size,] np.array of states
        mask:   [batch_size,ac_dim] np.array of masks
        done:   [batch_size,] binary np.array. 
            A 0 corresponds to a terminal game state, a 1 is a non-terminal game state
    '''
    
    #Make the easy lists
    observation_list = np.concatenate([path['observation'] for path in paths])
    action_list = np.concatenate([path['action'] for path in paths])
    reward_list = np.concatenate([path['reward'] for path in paths])
    mask_list = np.concatenate([path['mask'] for path in paths])

    #Make the done list
    number_of_states = len(observation_list)
    list_of_ones = [1] * number_of_states
    partial_sum =0
    for path in paths:
        partial_sum += len(path['observation'])
        list_of_ones[partial_sum-1] = 0
    done_list = list_of_ones
    

    #Select randomly chosen entries
    indices = np.random.choice(number_of_states, batch_size) 
    state1 = np.array([observation_list[i] for i in indices])
    action = np.array([action_list[i] for i in indices])
    state2 = np.array([observation_list[(i+1) % number_of_states] for i in indices])
    reward = np.array([reward_list[i] for i in indices])
    mask = np.array([mask_list[(i+1) % number_of_states] for i in indices])
    done = np.array([done_list[i] for i in indices])
    
    return state1, action, state2 , reward, mask, done

