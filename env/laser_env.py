import sys
import inspect
import os
from typing import Iterable

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import torch
from LaserModel import ComputationalLaser
from utils import *

import gymnasium as gym

def instantiate_laser(
        compressor_params:Iterable[float], 
        B_integral:float
        )->object: 
    """This function instantiates a Laser Model object based on a given parametrization (in terms of B integral and 
    compressor params).

    Returns:
        object: LaserModel-v2 object.
    """
    frequency, field = extract_data()  # extracts the data about input spectrum from data folde
    cutoff = np.array((289.95, 291.91)) * 1e12  # defines cutoff frequencies
    # cutting off the signal
    frequency_clean, field_clean = cutoff_signal(frequency_cutoff = cutoff, 
                                                 frequency = frequency * 1e12,
                                                 signal = field)
    # augmenting the signal (using splines)
    frequency_clean_aug, field_clean_aug = equidistant_points(frequency = frequency_clean,
                                                              signal = field_clean)  # n_points defaults to 5e3
    # retrieving central carrier
    central_carrier = central_frequency(frequency = frequency_clean_aug, signal = field_clean_aug)
    frequency, field = torch.from_numpy(frequency_clean_aug), torch.from_numpy(field_clean_aug)

    laser = ComputationalLaser(
        # laser specific parameters
        frequency = frequency * 1e-12, 
        field = field, 
        central_frequency=central_carrier,
        # environment parametrization
        compressor_params = compressor_params,
        B=B_integral)
    
    return laser

class LaserEnv(gym.Env): 
    """Custom gymnasium env for L1 Laser Pump"""
    def __init__(self, 
                 bounds:torch.tensor, 
                 compressor_params:torch.tensor,
                 B_integral:float, 
                 render_mode:str=None):
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

        self._bounds = bounds
        self._compressor_params = compressor_params
        self._B = B_integral
        self.render_mode = render_mode

        self._laser = instantiate_laser(
            compressor_params=self._compressor_params, 
            B_integral=self._B
        )
        # initial condition should be GDD, TOD and FOD of compressor with opposite sign
        self.GDD, self.TOD, self.FOD = -1 * self._compressor_params
        # STATES: FROG TRACE
        # make sure parameters_to_frog is a method of Computational Laser...
        self._observation = None
        # self._observation = parameters_to_frog_trace([self.GDD, self.TOD, self.FOD])

        # COMMENT: ACTIONS MIGHT BE CONSIDERED AS DELTAS FROM A GIVEN SET MASKING SOME VALUES OUT - THIS INTRODUCES 
        #          MASKING (WHICH MAKES THINGS HARDER) BUT CLEANS UP TRAINING PHASE. 
        #          OTHERWISE ONE COULD USE LARGELY NEGATIVE REWARD FOR TOO LARGE UPDATES.  
        # OPEN QUESTIONS: SINCE FROG TRACE DIRECTLY DERIVE FROM PARAMETERS, WHY NOT USING THE PARAMETERS THEMSELVES 
        #                 AS OBSERVATIONS?HOW WOULD YOU MASK IN CONTINOUS PROBLEMS? MAYBE WITH PADDING PROB. 
        #                 DISTRIBUTIONS?
        # ~: ASK GABRI
    
    @property
    def laser(self)->object:
        """Returns Laser object"""
        return self._laser
    
    @property
    def B(self)->float: 
        """Returns private value of B integral"""
        return self._B
    
    @property
    def compressor_params(self)->torch.tensor: 
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
    def update_compressor(self, new_params:torch.tensor)->None: 
        """Updates compressor parameters value with new set of values."""
        # updates env
        self._compressor_params = new_params
        # updates the simulator
        self._laser.compressor_params = new_params
    
    def _get_obs(self)->dict: 
        """Returns observation"""
        return self._observation
    
    def _get_info(self)->dict:
        """Returns info dicionary"""
        info = None  # for now, must be filled up with information about the phenomenon evolution

    def reset(self)->None: 
        """Resets the environment to initial observations"""
        self.GDD, self.TOD, self.FOD = -1 * self.compressor_params
        return self._get_obs() , self._get_info()
    
    def step(self, action:torch.tensor):
        """Applyies given action on laser env. Applying an action coincides with using it for
        controlling the actual laser."""

        # for now, does nothing and simply retrieves the temporal axis and profile given the present set
        # of parameters
        psi = torch.tensor([self.GDD, self.TOD, self.FOD])
        observation = self._laser.forward_pass(control=psi)
        
        reward = None
        done = False
        info = self._get_info()
        
        return observation, reward, done, info
    
    def render(self):
        # how to render?? Rendering is going to be super useful :))
        pass