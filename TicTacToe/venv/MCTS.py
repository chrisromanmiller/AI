import numpy as np

class mcts_node_:
    """
    self.observation:   current observation of environment
    self.actions:       current legal moves, n a format able to be fed into environment.step()
    self.rewards:       array of rewards obtained per action
    self.visit_N:       number of times this state has been seen
    self.action_visit_N:number of times each action has been picked from this state
    """

    def __init__(self, observation, legal_moves):
        """state should be whatever form using for the mcts dictionary
           actions should be a list of actions that can be fed into env.step"""
        self.observation = observation
        possible_actions = np.where(legal_moves)[0]
        self.actions = possible_actions
        self.rewards = np.zeros((len(self.actions),))
        self.visit_N = 0
        self.action_visit_N = np.zeros((len(self.actions),), dtype=np.int64)
        # self.action_children = {}

    def tree_policy(self):
        """Takes in a mcts_node
            -either picks random strategy or uses UCT1 to pick strategy
            returns the index for mcts_node.actions, NOT the action itself
            """
        unvisited_action_indices = np.where(self.action_visit_N == 0)[0]
        # Default policy: uniform random
        if len(unvisited_action_indices) > 0:
            return np.random.choice(unvisited_action_indices)
        else:
            UTC = self.get_UTC1()
            return np.argmax(UTC)

    def exploit_policy(self):
        """returns highest expected reward move among known moves, if no known moves random"""
        visited_action_indices = np.where(self.action_visit_N != 0)[0]
        # Default policy: uniform random
        if len(visited_action_indices) > 0:
            expected_reward = np.divide(self.rewards[visited_action_indices], self.action_visit_N[visited_action_indices])
            optimal_index = visited_action_indices[np.argmax(expected_reward)]
            return self.actions[optimal_index]
        else:
            return np.random.choice(self.actions)

    def get_UTC1(self, explore_constant = 1):
        return np.divide(self.rewards, self.action_visit_N) + explore_constant * np.sqrt(
                np.log(self.visit_N) * (1. / self.action_visit_N))

    def get_expected_rewards(self):
        return np.divide(self.rewards,self.action_visit_N)

    def __str__(self):
        return str({ "observation" : str(self.observation), "actions" : self.actions, "rewards" : self.rewards,
                 "visit_N" : self.visit_N, "action_visit_N" : self.action_visit_N})

class mcts():

    def __init__(self, min_reward = 0, max_reward = 1):
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
            player_id = environment.current_player
            observation = environment.get_observation(player_id)
            observation_hash = environment.hash_observation(observation)
            try:
                mcts_node_cur = self.states[observation_hash]
            except KeyError:
                mcts_node_cur = mcts_node_(observation, environment.legal_moves())
                self.states[observation_hash] = mcts_node_cur

            assert np.max(np.abs(np.diff(observation - mcts_node_cur.observation))) == 0, "hashing problem"

            next_action = mcts_node_cur.tree_policy()

            backpropogate_mcts_nodes.append(mcts_node_cur)
            backpropogate_actions.append(next_action)
            backpropogate_player_index.append(environment.current_player)

            environment.step(mcts_node_cur.actions[next_action])

            backpropogate_rewards.append(np.array(environment.rewards))

        #backpropogate reward
        rollout_N = len(backpropogate_rewards)
        players_N = len(backpropogate_rewards[0])
        backpropogate_cummulative_reward = np.zeros((rollout_N, players_N))
        #tracks the total reward
        total_reward = backpropogate_rewards[-1]
        for index in reversed(range(rollout_N)):
            backpropogate_cummulative_reward[index] = total_reward
            player_index = backpropogate_player_index[index] - 1

            if index > 0:
                total_reward[player_index] += backpropogate_rewards[index -1][player_index - 1]

        #upadte all visited nodes
        for mcts_node, action, reward_to_go, player_index in zip(backpropogate_mcts_nodes, backpropogate_actions, backpropogate_cummulative_reward, backpropogate_player_index):
            #update the node count, edge count, and edge reward
            mcts_node.rewards[action] += reward_to_go[player_index - 1]
            mcts_node.visit_N += 1
            mcts_node.action_visit_N[action] += 1










