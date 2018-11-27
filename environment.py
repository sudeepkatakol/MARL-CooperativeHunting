import numpy as np
import matplotlib.pyplot as plt

class ActionSpace():
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """
    def __init__(self):
        self.actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    def sample(self, seed=None):
        if seed is not None:
            return self.actions[np.random.RandomState(seed=seed).randint(low=0, high=len(self.actions))]
        return self.actions[np.random.randint(low=0, high=len(self.actions))]

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        return True if x in self.actions else False


class StateSpace():
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """
    def __init__(self):
        self.states = [(i, j) for j in range(1, 9) for i in range(1, 9)] ## Some are non reachable
    
    def sample(self, seed=None):
        if seed is not None:
            return np.random.RandomState(seed=seed).choice(self.states)
        return np.random.choice(self.states)

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        return True if x in self.states else False

class Enviroment():
    
    def __init__(self):
        #reward_range = (-np.inf, np.inf)
        self.action_space = ActionSpace()
        self.state_space = StateSpace() # state_space
        self.q_table = {}
        for st in self.state_space.states:
            self.q_table[st] = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.float32)
        self.info = {}
        self.t = 0
        self.start_state = (1, 2)
        self.current_state = (1, 2)
        self._seed = None
        self.agent_A_pos = (0.25,0.5)
        self.agent_B_pos = (2.75,0.5)
        self.status = 'train'
    
    def set_status(self,status):
        self.status = status
    
    def _next_state_A(self, state_A, action_A):
        up = {1:3, 3:6, 6:6, 4:7, 7:7, 2:5, 5:8, 8:8}
        right = {1:4, 3:4, 6:7, 4:5, 7:8, 2:8, 5:8, 8:8}
        if action_A == 1:
            return up[state_A]  
        else:
            return right[state_A]
    
    def _next_state_B(self, state_B, action_B):
        up = {1:3, 3:6, 6:6, 4:7, 7:7, 2:5, 5:8, 8:8}
        left = {1:1, 3:3, 6:6, 4:3, 7:6, 2:4, 5:4, 8:7}
        if action_B == 1:
            return up[state_B]  
        else:
            return left[state_B]
    
    def _is_final_state(self, new_state):
        assert(self.state_space.contains(new_state))
        ## Co operation: All 3 ways
        if new_state[0] == 7 and new_state[1] == 7:
            return True
        
        ## Fight!: All 2 ways
        if new_state[0] == 4 and new_state[1] == 4:
            return True
        
        ## A hunts B: Just one way this happens: Sightly tricky
        if new_state[0] == 7 and new_state[1] != 7:
            return True
        
        ## B hunts A: Just one way this happens: Slightly tricy
        if new_state[0] != 7 and new_state[1] == 7:
            return True
        
        return False
        
    def _reward(self, new_state):
        assert(self.state_space.contains(new_state))
        ## Co operation
        if new_state[0] == 7 and new_state[1] == 7:
            return (3, 3)
        
        if new_state[0] == 4 and new_state[1] == 4:
            return (-1, -1)
        
        if new_state[0] == 7 and new_state[1] != 7:
            return (2, 0)
        
        if new_state[0] != 7 and new_state[1] == 7:
            return (0, 2)
        return (0, 0)
    
    def make_grid(self):
        x = np.arange(0, 4, 1)
        y = np.arange(0, 4, 1)
        
        fig = plt.figure(figsize=(6, 5))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        plt.grid()
        agent_A = plt.Circle(self.agent_A_pos,0.1,color='r')
        agent_B = plt.Circle(self.agent_B_pos,0.1,color='b')
        prey = plt.Circle((1.5,2.5),0.1,color='g')
        ax.add_artist(agent_A)
        ax.add_artist(agent_B)
        ax.add_artist(prey)
        plt.savefig("./images/" + str(self.t) + ".png")
    
    def change_pos(self, new_state):
        #For Agent A
        _A = {1:(0.25,0.5), 3:(0.25,1.5), 6:(0.25,2.5), 
              4:(1.25,1.5), 7:(1.25,2.5),
              2:(2.25,0.5), 5:(2.25,1.5), 8:(2.25,2.5)}
        _B = {1:(0.75,0.5), 3:(0.75,1.5), 6:(0.75,2.5), 
              4:(1.75,1.5), 7:(1.75,2.5),
              2:(2.75,0.5), 5:(2.75,1.5), 8:(2.75,2.5)}
        self.agent_B_pos = _B[new_state[1]]
        self.agent_A_pos = _A[new_state[0]]
    
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        new_state = (self._next_state_A(self.current_state[0], action[0]),                     self._next_state_B(self.current_state[1], action[1]))
        done = self._is_final_state(new_state)
        reward = self._reward(new_state)
        self.t += 1
        self.info[self.t] = (new_state, reward, done)
        self.current_state = new_state
        self.change_pos(new_state)
        self.render()
        return new_state, reward, done, self.info[self.t]

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.agent_A_pos = (0.25,0.5)
        self.agent_B_pos = (2.75,0.5)
        self.render()
        self.t = 0
        self.current_state = self.start_state
        self.render()
        return self.current_state

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        if(self.status =='train'):
            pass
        elif(self.status =='gui'):
            self.make_grid()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        _seed = seed
        return _seed

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    #def __del__(self):
    #    self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
