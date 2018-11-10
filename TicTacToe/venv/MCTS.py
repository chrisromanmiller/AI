import numpy as np
import copy

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
        import copy

        self.observation = copy.copy(observation)
        possible_actions = np.where(legal_moves)[0]
        self.actions = possible_actions
        self.rewards = np.zeros((len(self.actions),))
        self.visit_N = 0
        self.action_visit_N = np.zeros((len(self.actions),), dtype=np.int64)
        # self.action_children = {}

    def tree_policy(self, exploration_constant = 1):
        """Takes in a mcts_node
            -either picks random strategy or uses UCT1 to pick strategy
            returns the index for mcts_node.actions, NOT the action itself
            """
        unvisited_action_indices = np.where(self.action_visit_N == 0)[0]
        # Default policy: uniform random
        if len(unvisited_action_indices) > 0:
            return np.random.choice(unvisited_action_indices)
        else:
            UTC = self.get_UTC1(exploration_constant)
            return np.argmax(UTC)

    def exploit_policy(self):
        """returns highest expected reward move among known moves, if no known moves random"""
        visited_action_indices = np.where(self.action_visit_N != 0)[0]
        # Default policy: uniform random
        if len(visited_action_indices) > 0:
            # expected_reward = np.divide(self.rewards[visited_action_indices], self.action_visit_N[visited_action_indices])
            optimal_index = np.argmax(self.action_visit_N)
            return self.actions[optimal_index]
        else:
            return np.random.choice(self.actions)

    def get_UTC1(self, exploration_constant = 1):
        return np.divide(self.rewards, self.action_visit_N) + exploration_constant * np.sqrt(
                np.log(self.visit_N) * (1. / self.action_visit_N))

    def get_expected_rewards(self):
        return np.divide(self.rewards,self.action_visit_N)

    def __str__(self):
        return str({ "observation" : str(self.observation), "actions" : self.actions, "rewards" : self.rewards,
                 "visit_N" : self.visit_N, "action_visit_N" : self.action_visit_N})

class mcts():

    def __init__(self, exploration_costant = 1):
        """We want an environment with the following commands:
            env.legal_moves()
            env.get_observation()
            env.get_reward()
            env.step()
            """

        self.states = {}
        self.exploration_constant = exploration_costant
        # self.environment_rollout = copy.deepcopy(self.environment)

    def batch_rollout(self, environment, total_simulations):
        simulation_N = 0
        while simulation_N < total_simulations:
            simulation_N += self.rollout(environment)

    def grab_mcts_node(self, environment):
        observation = environment.get_observation()
        observation_hash = environment.hash_observation(observation)
        try:
            mcts_node_cur = self.states[observation_hash]
        except KeyError:
            mcts_node_cur = mcts_node_(observation, environment.legal_moves())
            self.states[observation_hash] = mcts_node_cur

        assert (not np.any(np.logical_xor(observation, mcts_node_cur.observation))), "hashing problem"

        return mcts_node_cur


    def rollout(self, environment):

        import copy
        environment_rollout = copy.deepcopy(environment)

        backpropogate_mcts_nodes = []
        backpropogate_actions = []
        backpropogate_rewards = []
        backpropogate_player_index = []

        #rollout phase, using tree policy
        while not environment_rollout.done:

            mcts_node_cur = self.grab_mcts_node(environment_rollout)

            next_action = mcts_node_cur.tree_policy(exploration_constant=self.exploration_constant)

            backpropogate_mcts_nodes.append(mcts_node_cur)
            backpropogate_actions.append(next_action)
            backpropogate_player_index.append(environment_rollout.current_player)

            environment_rollout.step(mcts_node_cur.actions[next_action])

            backpropogate_rewards.append(np.array(environment_rollout.rewards))

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

        #upadte all visited nodes
        for mcts_node, action, reward_to_go, player_index in zip(backpropogate_mcts_nodes, backpropogate_actions, backpropogate_cummulative_reward, backpropogate_player_index):
            #update the node count, edge count, and edge reward
            mcts_node.rewards[action] += reward_to_go[player_index]
            mcts_node.visit_N += 1
            mcts_node.action_visit_N[action] += 1

        return rollout_N








