import torch
import gym
from .env_utils import instantiate_laser
from utils import *

class Abstract_BaseLaser(gym.Env):
    """
    Custom gymnasium env for L1 Laser Pump. 
    This class abstracts actions and observation space.
    """
    def __init__(self, 
                 bounds:torch.TensorType, 
                 compressor_params:torch.TensorType,
                 B_integral:float, 
                 render_mode:str=None, 
                 seed:int=None):
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
        """
        # environment parametrization
        self._bounds = bounds
        self._compressor_params = compressor_params
        self._B = B_integral
        # render mode
        self.render_mode = render_mode
        # set the actual laser
        self._laser = instantiate_laser(
            compressor_params=self._compressor_params, 
            B_integral=self._B
        )
        # abstracts observation and action space
        self._observation = None
        self._observation_space = None
        self.action_space = None
        self._seed = seed
    
    @property
    def tensor_observation(self): 
        return self._observation
        # return torch.from_numpy(self._observation)

    @property
    def laser(self)->object:
        """Returns Laser object"""
        return self._laser
    
    @property
    def B(self)->float: 
        """Returns private value of B integral"""
        return self._B
    
    @property
    def compressor_params(self)->torch.TensorType: 
        """Returns compressor params for laser"""
        return self._compressor_params

    @B.setter
    def update_B(self, new_B:int)->None: 
        """Updates the value of B integral. When updating, also updates the laser changing the value
        of laser's B."""

        if not new_B > 0:
            raise ValueError(f"B integral must be > 0! Prompted {new_B}")
        # updates env
        self._B = new_B
        # updates the simulator
        self._laser.B = new_B

    @compressor_params.setter
    def update_compressor(self, new_params:torch.TensorType)->None: 
        """Updates compressor parameters value with new set of values."""
        # updates env
        self._compressor_params = new_params
        # updates the simulator
        self._laser.compressor_params = new_params
    
    def _get_obs(self)->dict: 
        """Returns observation"""
        return self._observation
    
    def _get_info(self)->dict:
        """Returns info dicionary. Info's should be drawned from current observation."""
        pass

    def reset(self)->None: 
        """Resets to initial conditions."""
        pass
    
    def step(self, action:torch.TensorType):
        """Updates the observation based on action."""
        pass
    
    def render(self):
        """Renders current state."""
        pass

    def seed(self, seed:int):
        self._seed = seed