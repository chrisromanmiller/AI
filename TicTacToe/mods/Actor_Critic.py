#A module the contains the functions for the Actor-Critic algorithm
import tensorflow as tf
import numpy as np
from collections import Counter


tf.reset_default_graph()

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
        out = tf.layers.dense(out, num_actions , kernel_initializer = tf.ones_initializer(), bias_initializer = tf.zeros_initializer(), activation = tf.nn.sigmoid)
        out = (out*2)-1
    return out

    
# def sample_action(model, mask_placeholder):
#     '''Symbolically selects an action from logits with restrictions
#     Inputs: 
#         model: a [None, action_dim] variable consisting of logits
#         mask_placeholder: a [None, action_dim] placeholder that will be fed boolean vectors
    
#     Returns:
#         A random legal action, where legal values are those which mask_placeholder assigns 1
#         The probabilities are weighted according to the logits
#     '''
#     out = model
#     dist = tf.distributions.Categorical(probs=maskedSoftmax(out, mask_placeholder))
#     return dist.sample()
    


def policy_distribution(model):
    '''Symbolically selects an action from logits with restrictions
    Inputs: 
        model: a [None, action_dim] variable consisting of logits
        mask_placeholder: a [None, action_dim] placeholder that will be fed boolean vectors
    
    Returns:
        A random legal action, where legal values are those which mask_placeholder assigns 1
        The probabilities are weighted according to the logits
    '''
    out = model
    dist = tf.nn.softmax(out)
    return dist




    
def maskedSoftmax(logits, mask):
    '''Computes the softmax of our logits, given that some moves are illegal
    Inputs:
        Masked softmax over dim 1
        param logits: [None, ac_dim]
        param mask: [None, ac_dim]
        
        ***This code is edited from code we found online***
        We do not want there to be any probability of making illegal moves. 
        Intuitively, we are computing softmax of our logits, but pretending that the only entries 
            are the legal ones.
        This is actually implemented via SparseTensor calculations.
        
    Returns: 
        result: [None, ac_dim] a sequence of probability distributions, with zero probability of illegal moves
    '''
    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    
    # Tensorflow will automatically set output probabilities to zero of 
    # undesignated entries in sparse vector
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    
    result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)
    return result


def batch_rollout(player,opponent, env, max_time_steps = 100):
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
    batch_winners = Counter({0: 0, 1: 0, 2:0})
    time_steps = 0
    while time_steps < max_time_steps:
        path = sample_trajectory(player,opponent,env)
        paths += [path]
        batch_winners[env.current_winner] +=1
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
    
    #Do rest of moves
    while not done:
        #Get current observation of current player
        ob = env.get_observation(env.current_player)
        legal_moves = env.legal_moves()
        if env.current_player == 1:
            #Reward is recorded as results of state,action pair... need to check player 1 has acted already
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

    #Need to record final reward for player 1
    rewards.append(env.get_reward(1))
    
    path = {"observation" : np.array(obs, dtype=np.int32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.int32),
                "mask" : np.array(masks, dtype=np.int32)}
    return path

    
    
def sum_of_rewards(paths, gamma = .6): 
    re_n = [path["reward"] for path in paths]
    q_n = []
    for seq_of_rewards in re_n:
        for t in range(len(seq_of_rewards)):
            weighted_sequence = seq_of_rewards[t:] * np.array([gamma**i for i in range(len(seq_of_rewards[t:]))])
            q_n.append(np.sum(weighted_sequence))
    adv_n = q_n
    return adv_n
        
def standardize_advantage(adv_n):
    adv_n = (adv_n - np.mean(adv_n)) 
    adv_n = adv_n * (1.0/(np.std(adv_n)+.0000001))
    return adv_n

def get_log_prob(model, action_placeholder, mask_placeholder):
    action_dim = 9 
    logits = model
    
    indices = tf.where(mask_placeholder)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    
    """THIS IS THE KEY: tensorflow will automatically set output probabilities to zero of undesignated entries in sparse vector"""
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    
    probability_dist = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
#     probability_dist = probability_dist.set_shape(logits.shape)
    log_probability_dist = tf.scatter_nd(sparseResult.indices, tf.log(sparseResult.values), sparseResult.dense_shape)

    """Want to emulate this:"""
#     probability_dist = tf.nn.softmax(logits)
#     legal_pseudo_probability_dist = probability_dist*values
#     legalprobability_dist = tf.divide(legal_pseudo_probability_dist, tf.reduce_sum(legal_pseudo_probability_dist, axis= 1))
    
    prod = tf.multiply(probability_dist, tf.one_hot(action_placeholder, action_dim ))
    
    entropy = - tf.reduce_sum(probability_dist * log_probability_dist, axis = 1)
    
    
    
    log_prob = tf.log(tf.reduce_sum(prod , axis = 1 ))
#    log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels= action_placeholder, logits= tf.SparseTensor(indices, values, denseShape))
    return log_prob, entropy

def loss_and_update_op(log_prob, entropy, adv_n, entropy_coeff = .1):
    loss = -tf.reduce_mean(log_prob * adv_n) -  entropy_coeff * entropy
    optimizer = tf.train.AdamOptimizer(5e-3)
    update_op = optimizer.minimize(loss)
    return loss, update_op
    