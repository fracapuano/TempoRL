import sys
from typing import Iterable, Tuple
import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from .env_utils.LaserModel_wrapper import *

import gym
from gym import spaces

class LaserEnv(gym.Env):
    """Custom Environment implementing Laser Model that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        a:int = -5, 
        b:int = 5, 
        action_shape:Tuple[float, ] = (3,), 
        obs_shape:Tuple[float, ] = (16,),
        number_buildups:float = 100.,
        number_FW:float = 100., 
        maximal_timesteps:int = 100) -> None:
        """This function initializes the environment. Input to this function are the lower (a) and upper (b) fictional bounds
        to express action space.
        """
        super(LaserEnv, self).__init__()
        self.a, self.b = a, b
        self.action_shape = action_shape
        self.obs_shape = obs_shape

        self.LaserWrapper = LaserWrapper(a = self.a, b = self.b)
        
        # defining continous action space
        self.action_space = spaces.Box(low = self.a, high = self.b, shape = self.action_shape)
        
        # defining continous observation space
        self.observation_space = spaces.Box(
            low = np.full(self.obs_shape, -float('inf'), dtype=np.float32), 
            high = np.full(self.obs_shape, +float('inf'), dtype=np.float32),
            shape = self.obs_shape
            )
        
        # initial state
        self.state = self.random_state()

        # transform limited characteristics
        self.TL_embedding = self.LaserWrapper.transform_limited_embedding()

        # TERMINATION CONDITIONS 
        # 100 times the TL buildup
        self.buildup_max = number_buildups * self.TL_embedding["Buildup duration"]
        # 100 times the TL FWHM
        self.FWHM_max = number_FW * self.TL_embedding["FWHM"]
        # maximal number of timesteps per episode
        self.maximal_timesteps = maximal_timesteps
        self.remaining_steps = maximal_timesteps

        # States and Actions Log
        self.visited_states = []
        self.applied_actions = []
        self.done = False
    
    def step(self, action:np.array)->Tuple[np.array, float, bool, dict]:
        # converting action to tensor
        action = torch.from_numpy(action)
        # storing current state - column vectors
        self.visited_states.append(self.state.reshape(-1,))
        # applying action and updating the state
        self.state = self.LaserWrapper.forward_pass(action)
        # storing applied action - column vectors
        self.applied_actions.append(action.reshape(-1,))
        # reduce remaining timesteps
        self.remaining_steps -= 1
        # evaluate completion at current state
        self.done = self.evaluate_completion()
        # obtain reward
        if self.done: 
            info = {"Buildup condition": self.buildup_condition, "FWHM condition": self.FWHM_condition, "Available Timesteps": self.remaining_steps}
            if self.remaining_steps <= 0: 
                reward = 0
            else:
                reward = -5 # penalization for failing
        
        elif not self.done: 
            reward_peak, penalty_action = self.compute_reward()
            info = {"Reward Peak": reward_peak, "Penalty Action": penalty_action, "Info": self.remaining_steps}
            alive_bonus = 10
            reward = 100 * reward_peak + alive_bonus - penalty_action
            reward = reward.item()
        # return obs, reward, done and info
        return self.state, reward, self.done, info

    def compute_reward(self)->Tuple[float, float]: 
        """This function evaluates the current state and returns elements of the reward function. 

        Returns:
            Tuple[float, float]: Peak Intensity Reward, Area Reward
        """
        transform_limited_I0 = self.TL_embedding["Peak Intensity"]
        if len(self.applied_actions) == 1: 
            penalty_action = 0
        else: 
            previous_action, last_action = self.applied_actions[-2:]
            penalty_action = (((last_action - previous_action) / previous_action) ** 2).sum()
        # Peak Intensity: 0
        reward_peak = self.state[0] / transform_limited_I0
        
        return reward_peak, penalty_action

    def evaluate_completion(self)->bool: 
        """This function evaluates an input state to conclude whether or not completion is observed at current state.

        Returns:
            bool: Whether or not the episode is complete. 
        """
        # convert state to indexed series (for code readibility)
        state_series = pd.Series(data = self.state, index = self.TL_embedding.index)
        
        self.buildup_condition = state_series["Buildup duration"] >= self.buildup_max
        self.FWHM_condition = state_series["FWHM"] >= self.FWHM_max

        if self.buildup_condition or self.FWHM_condition or self.remaining_steps <= 0:
            return True
        else: 
            return False

    def random_state(self): 
        """This function draws a random control from the action state to initialize the state.
        """
        random_action = self.action_space.sample()
        return self.LaserWrapper.forward_pass(control = torch.from_numpy(random_action))

    def reset(self):
        """This function resets the env when failing is observed.
        """
        self.done = False
        self.remaining_steps = self.maximal_timesteps
        self.state = self.random_state()
        self.applied_actions = []
        self.visited_states = []
        return self.state  # reward, done, info can't be included

    def convert_actions(self)->Iterable: 
        """This function converts the applied actions into actual controls that can be used in LaserModel. 
        """
        return map(
            lambda control: descale_control(control = control, given_bounds = self.LaserWrapper.bounds_control, actual_bounds = self.LaserWrapper.bounds_SI), 
            self.applied_actions
        )
    
    def convert_states(self)->Iterable: 
        """This function converts the visited states so that it is possible to easy interpret them
        """
        return map(lambda s: descale_embedding(s), self.visited_states)
    
    def convert_TL_embedding(self)->pd.Series: 
        """This function converts the TL embedding considered
        """
        return pd.Series(data = descale_embedding(self.TL_embedding.values), index = self.TL_embedding.index)

    def convert_obs(self, state:np.array)->np.array: 
        """This function converts an input state to descale it
        """
        return descale_embedding(state)
