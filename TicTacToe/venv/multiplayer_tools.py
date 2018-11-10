
from collections import Counter
import numpy as np


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
    done = False
    player_has_acted = False

    # Do rest of moves
    while not done:
        # Get current observation of current player
        ob = env.get_observation()
        legal_moves = env.legal_moves()
        environments = [env]
        if env.current_player == 1:
            # Reward is recorded as results of state,action pair... need to check player 1 has acted already
            if player_has_acted:
                rewards.append(env.rewards[0])
            else:
                player_has_acted = True

            action = player.policy(environments)
            obs.append(ob)
            acs.append(action[0])
            masks.append(legal_moves)
        else:
            action = opponent.policy(environments)
        done, _ = env.step(action[0])

        # Need to record final reward for player 1
    rewards.append(env.rewards[0])

    path = {"observation": np.array(obs, dtype=np.bool_),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.int32),
            "mask": np.array(masks, dtype=np.bool_)}
    return path


def sample_paths(paths, batch_size=10):
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

    # Make the easy lists
    observation_list = np.concatenate([path['observation'] for path in paths])
    action_list = np.concatenate([path['action'] for path in paths])
    reward_list = np.concatenate([path['reward'] for path in paths])
    mask_list = np.concatenate([path['mask'] for path in paths])

    # Make the done list
    number_of_states = len(observation_list)
    list_of_ones = [1] * number_of_states
    partial_sum = 0
    for path in paths:
        partial_sum += len(path['observation'])
        list_of_ones[partial_sum - 1] = 0
    done_list = list_of_ones

    # Select randomly chosen entries
    indices = np.random.choice(number_of_states, batch_size)
    state1 = np.array([observation_list[i] for i in indices])
    action = np.array([action_list[i] for i in indices])
    state2 = np.array([observation_list[(i + 1) % number_of_states] for i in indices])
    reward = np.array([reward_list[i] for i in indices])
    mask = np.array([mask_list[(i + 1) % number_of_states] for i in indices])
    done = np.array([done_list[i] for i in indices])

    return state1, action, state2, reward, mask, done

