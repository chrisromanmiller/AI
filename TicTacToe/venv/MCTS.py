import numpy as np


class mcts_node():
    """
    self.observation:   current observation of environment
    self.actions:       current legal moves, n a format able to be fed into environment.step()
    self.rewards:       array of rewards obtained per action
    self.visit_N:       number of times this state has been seen
    self.action_visit_N:number of times each action has been picked from this state
    """

    def __init__(self, observation, actions):
        """state should be whatever form using for the mcts dictionary
           actions should be a list of actions that can be fed into env.step"""
        self.observation = observation
        self.actions = actions
        self.rewards = np.zeros((len(actions),))
        self.visit_N = 0
        self.action_visit_N = np.zeros((len(actions),), dtype=np.int64)
        # self.action_children = {}



class mcts():

    def __init__(self, min_reward = -1, max_reward = 1):
        """We want an environment with the following commands:
            env.legal_moves()
            env.get_observation()
            env.get_reward()
            env.step()
            """

        self.states = {}
        self.min_reward = min_reward
        self.max_reward = max_reward


    def rollout(self, environment_original):

        import copy
        environment = copy.deepcopy(environment_original)

        backpropogate_mcts_nodes = []
        backpropogate_actions = []
        backpropogate_rewards = []
        backpropogate_player_index = []

        #rollout phase, using tree policy
        while not environment.done:
            mcts_node = None
            player_id = environment.current_player
            observation = environment.get_observation(player_id)
            try:
                mcts_node = self.states[environment.observation]
            except KeyError:
                possible_actions = np.where(environment.legal_moves())[0]
                mcts_node = mcts_node(observation, possible_actions)

            next_action = self.tree_policy(mcts_node)

            backpropogate_mcts_nodes.append(mcts_node)
            backpropogate_actions.append(next_action)
            backpropogate_player_index.append(environment.current_player)

            environment.step(mcts_node.actions[next_action])

            backpropogate_rewards.append((np.array(environment.get_rewards()) - self.min_reward)/ (self.max_reward-self.min_reward))

        #backpropogate reward
        rollout_N = len(backpropogate_rewards)
        players_N = len(backpropogate_rewards[0])
        backpropogate_cummulative_reward = np.zeros((rollout_N, players_N))
        #tracks the total reward
        total_reward = backpropogate_rewards[-1]
        for index in reversed(range(rollout_N)):
            backpropogate_cummulative_reward[index] = total_reward
            player_index = backpropogate_player_index[index]

            if index > 0:
                total_reward[player_index] += backpropogate_rewards[index -1][player_index]

        #backpropogation phase
        for mcts_node, action, reward_to_go, player_index in zip(backpropogate_mcts_nodes, backpropogate_actions, backpropogate_cummulative_reward, backpropogate_player_index):
            #update the node count, edge count, and edge reward
            mcts_node.rewards[action] += reward_to_go[player_index]
            mcts_node.visit_N += 1
            mcts_node.action_visit_N[action] += 1









    def tree_policy(self, mcts_node):
        """Takes in a mcts_node
            -either picks random strategy or uses UCT1 to pick strategy
            returns the index for mcts_node.actions, NOT the action itself
            """
        unvisited_action_indices = np.where(mcts_node.action_visit_N == 0)[0]




        # Default policy: uniform random
        if len(unvisited_action_indices) > 0:
            return np.random.choice(unvisited_action_indices)
        else:
            UTC = np.divide(mcts_node.rewards, mcts_node.action_visit_N) + 0.25 * np.sqrt(
                np.log(mcts_node.visit_N) * np.reciprocal(mcts_node.action_visit_N))
            return mcts_node.actions[np.argmax(UTC)]



