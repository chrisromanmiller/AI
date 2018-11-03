import numpy as np

class mcts():

     class mcts_state():

         def __init__(self, state, actions):
             """state should be whatever form using for the mcts dictionary
                actions should be a list of actions that can be fed into env.step"""
             self.state = state
             self.actions = actions
             self.rewards =  np.zeros((len(actions),))
             self.visit_N = 0
             self.action_visit_N = np.zeros((len(actions),),dtype=np.int64)

        def tree_policy(self):
            unvisited_action_indices = np.where(self.action_visit_N == 0)[0]

            #Default policy: uniform random
            if len(unvisited_action_indices) > 0:
                return self.actions[np.random.choice(unvisited_action_indices)]
            else:
                UTC = np.divide(self.rewards, self.action_visit_N) + 0.25*np.sqrt(np.log(self.visit_N)*np.reciprocal(self.action_visit_N) )
                return self.actions[np.argmax(UTC)]





    def __init__(self, environment):
        """We want an environment with the following commands:
            env.legal_moves()
            env.get_observation()
            env.get_reward()
            env.step()
            """
        self.environment = environment


