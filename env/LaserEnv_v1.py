import torch
from typing import Tuple, List
from gymnasium.spaces import Box

from .BaseLaser import Abstract_BaseLaser
from .env_utils import ControlUtils
from utils import physics

L1Loss = torch.nn.L1Loss(reduction="sum")

class LaserEnv_v1(Abstract_BaseLaser):
    """Instances a first version of the L1 Pump laser.
    
    In this version, observations are actual control parameters (that is, \psi). 
    Actions are deltas on said configurations (changes made on \psi). 
    The reward here depends both on the loss with respect to the target shape and the 
    magnitude of the actions. This biases the policy against rapsodic policies.
    """
    def __init__(
    self,
    bounds:torch.TensorType,
    compressor_params:torch.TensorType,
    B_integral:float,
    render_mode:str=None, 
    default_target:Tuple[bool, List[torch.TensorType]]=True)->None:
        """Init function. Here laser-oriented characteristics are defined.
        Args:
            bounds (torch.tensor): GDD, TOD and FOD upper and lower bounds. Shape must be (3x2). Values must be
                                    expressed in SI units (i.e., s^2, s^3 and s^4).
            compressor_params (torch.tensor): \alpha_{GDD}, \alpha_{TOD}, \alpha_{FOD} of laser compressor. If no 
                                                non-linear effects were to take place, one would have that optimal control
                                                parameters would be exactly equal to -compressor_params.
            B_integral (float): B_integral value. This parameter models non-linear phase accumulation. The larger, 
                                the higher the non-linearity introduced in the model.
            render_mode (str, optional): Render mode. Defaults to None.
            default_target (Tuple[bool, List[torch.TensorType], optional): Whether or not to use the default (transform-limited
                                                                           target).When not True, accepts target time axis and temporal 
                                                                           pulse.
        """
        # params init
        super().__init__(
             bounds=bounds, 
             compressor_params=compressor_params, 
             B_integral=B_integral
        )
        # control utils - suite to handle with ease normalization of control params
        self.control_utils = ControlUtils()  # initialized with default parameters 

        # render mode setted
        self.render_mode = render_mode
        # specifiying obs space
        self._observation_space = Box(
             low = torch.zeros(3).numpy(), 
             high = torch.ones(3).numpy()
        )
        # starting with random parameters
        self._observation = torch.from_numpy(self._observation_space.sample())

        # actions are defined as deltas 
        self.action_space = Box(
            low = -1 * torch.ones(3).numpy(), 
            high = +1 * torch.ones(3).numpy()
        )
        self.nsteps = 0  # number of steps to converge
        if default_target is True:
            self.target_time, self.target_pulse = self.laser.transform_limited()
        else: 
            self.target_time, self.target_pulse = default_target
        
        # defining maximal number of steps and value for sum(L1) loss 
        self.MAX_LOSS = 200  # pretty huge number, considering that pulses are in the always in the 0-1 range
        self.MAX_STEPS = 500
    
    def get_observation_SI(self):
        """
        Returns observation in SI units. SI-units observation are accepted as inputs to
        the Computational Laser considered.
        """
        return self.control_utils.remagnify_descale(self._observation)
    
    def _get_control_loss(self)->float:
        """This function returns the value of the L1-Loss for a given set of control parameters."""
        # obtain the pulse shape corresponding to given set of control params
        _, time, control_shape = self.laser.forward_pass(self.control_utils.remagnify_descale(self._observation))
        # move target and controlled pulse peak on peak
        pulse1, pulse2 = physics.peak_on_peak(
             temporal_profile=[time, control_shape], 
             other=[self.target_time, self.target_pulse]
             )
        # compute sum(L1 loss)
        return L1Loss(pulse1[1], pulse2[1]).item()

    def _get_info(self): 
        """Return state-related info."""
        info = {
            "current_control": self._observation,
            "distance_from_lower": torch.norm(self._observation).item(),
            "distance_from_upper": torch.norm(+1 * torch.ones(3) - self._observation).item(),
            "L1Loss": self._get_control_loss()
        }
        return info

    def reset(self)->None: 
        """Resets the environment to initial observations"""
        self._observation = torch.from_numpy(self._observation_space.sample())
        self.n_steps = 0

        return self._get_obs() , self._get_info()

    def is_done(self)->bool:
        """
        This function returns a boolean that represents whether or not the optimization process is done
        given the current state.
        """
        if self.n_steps >= self.MAX_STEPS or self._get_control_loss() >= self.MAX_LOSS: 
            return True  # stop episode, restart
        else:
            return False  # continue, episode not done yet

    def compute_reward(self, state:torch.TensorType, action:torch.TensorType)->float:
        """
        This function computes the reward associated with the (state, action) pair. 
        This reward function is made up of several different components and derives from fundamental assumptions
        made on the values that each term can take on.

        Args: 
            state (torch.TensorType): Current observation. Used to determine the temporal profile.
            action (torch.TensorType): Current action. Used to update the current state.
        
        Returns: 
            float: Value of reward. Sum of three different components. Namely: Error-Component, Action-Magnitude-Component, Number-of-Steps 
                   Component.
        """
        alive_bonus = self.n_steps / self.MAX_STEPS
        loss_penalty = -self._get_control_loss()/self.MAX_LOSS  # 0-1 range, in absolute value
        action_penalty = (-torch.norm(torch.from_numpy(action)))   # 0-1 range as well
        # this reward is large when loss is small and action performed is minimal
        return alive_bonus + loss_penalty + action_penalty

    def step(self, action:torch.TensorType):
        """
        Applyies given action on laser env. 
        """
        # increment number of steps
        self.n_steps += 1
        # applying action, clipping on +1 and -1
        self._observation = torch.clip(self._observation + torch.from_numpy(action), min=torch.zeros(3), max=torch.ones(3))
        
        reward = self.compute_reward(state=self._observation, action=action)
        done = self.is_done()
        info = self._get_info()
        
        return self._observation, reward, done, info