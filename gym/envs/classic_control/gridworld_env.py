import math
import copy
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class GridWorldEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs) :
        self.action_space=[0,1,2,3,4,5] #RIGHT, DOWN, LEFT, UP
        self.observation_space = spaces.Box(low=0.0, high=float(kwargs.get('dim',4)), shape=(2,))
        self.seed()
        self.viewer = None
        self._s = None
        self.steps_beyond_done = None
        # Action indices: RIGHT, DOWN, LEFT, UP, JUMP 4 cells DOWN (only for A'), JUMP 2 cells DOWN (only for B')
        self.action_effects = [ (1.0,0.0), (0.0,1.0), (-1.0,0.0), (0.0,-1.0),(0.0,4.0),(0.0,2.0)]
        self.terminals = [ np.array([ self.observation_space.low[0], self.observation_space.high[1], \
            self.observation_space.high[0], self.observation_space.low[1] ]) ]
        self.A = np.array([1, 0])
        self.B = np.array([3, 0])
        self.A_ = np.array([1, 4])
        self.B_ = np.array([3, 2])
        self.grid = np.zeros((5,5))
        self.steps = 0
        self.steps_limit = 15000


    def seed( self, seed=None ) :
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self) :
        return self._s

    @state.setter
    def state(self, v) :
        self._s = copy.deepcopy(v)


    def step(self, action) :
        x,y = self._s
        dx, dy = self.action_effects[action]
        next_x = x + dx
        next_y = y + dy
        
        self.steps += 1

        done = False

        # Wraparound
        if self.observation_space.low[0] > next_x or self.observation_space.high[0] < next_x or self.observation_space.low[1] > next_y or self.observation_space.high[1] < next_y :
            next_x = x
            next_y = y
            reward = -1.0
        else:
            reward = 0.0

        self.state = np.array([next_x, next_y])


        if np.array_equal(np.array([x,y]), self.A) and np.array_equal(self.state, self.A_):
            reward = 10.0
        elif np.array_equal(np.array([x,y]), self.B) and np.array_equal(self.state, self.B_):
            reward = 5.0

        if self.steps > self.steps_limit:
            done = True

        return self.state, reward, done, {}

    def reset(self, s = None) :
        if s is None :
            self._s = np.array([0.0, 0.0])
        else :
            self._s = copy.deepcopy(s)
        self.steps_beyond_done = None
        self.steps = 0
        return self._s

    def render(self, mode='human'):
        self.grid[self.state[1].astype(int), self.state[0].astype(int)] = 1
        print(self.grid,'\n')
        if self.steps > self.steps_limit:
            print("=======================================")
        self.grid = np.zeros((5,5))


    def close(self):
        if self.viewer: self.viewer.close()
