import torch
import numpy as np
from typing import Tuple, List
from gym.spaces import Box
from torch.distributions.multivariate_normal import MultivariateNormal

from .BaseLaser import Abstract_BaseLaser
from .env_utils import ControlUtils
from utils import physics
from utils.render import visualize_pulses
import pygame

import matplotlib.pyplot as plt
from PIL import Image

L1Loss = torch.nn.L1Loss(reduction="sum")
device = "cuda" if torch.cuda.is_available() else "cpu"

class LaserEnv_v1(Abstract_BaseLaser):
    metadata = {
        "render_fps":15, 
        "render_modes": ["rbg_array", "human"]
        }
    
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
    render_mode:str="rgb_array", 
    default_target:Tuple[bool, List[torch.TensorType]]=True, 
    init_variance:float=.1,
    device:str=device)->None:
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
        # state and action space dimensionality
        self.StateDim = 3
        self.ActionDim = 3

        # params init
        super().__init__(
             bounds=bounds, 
             compressor_params=compressor_params, 
             B_integral=B_integral, 
             render_mode=render_mode
        )
        # control utils - suite to handle with ease normalization of control params
        self.control_utils = ControlUtils()  # initialized with default parameters 

        # specifiying obs space
        self.observation_space = Box(
             low = np.zeros(self.StateDim), 
             high = np.ones(self.StateDim)
        )
        
        # actions are defined as deltas - updates are bounded
        self.action_space = Box(
            low = -0.2 * np.ones(self.ActionDim), 
            high = 0.2 * np.ones(self.ActionDim)
        )

        self.nsteps = 0  # number of steps to converge
        if default_target is True:
            self.target_time, self.target_pulse = self.laser.transform_limited()
        else: 
            self.target_time, self.target_pulse = default_target
        
        # defining maximal number of steps and value for aligned-sum(L1) loss 
        self.MAX_LOSS = 500
        self.MAX_STEPS = 500

        self.rho_zero = MultivariateNormal(
            # loc is compressor params (in the 0-1 range)
            loc=self.control_utils.demagnify_scale(-1 * self.compressor_params).float(), 
            # homoscedastic distribution
            covariance_matrix=torch.diag(init_variance * torch.ones(self.StateDim))
        )

        # starting with random parameters
        self._observation = torch.clip(self.rho_zero.sample(), torch.zeros(self.StateDim), torch.ones(self.StateDim))

    def get_observation_SI(self):
        """
        Returns observation in SI units. 
        SI-units observations only are accepted as inputs to the ComputationalLaser considered.
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
        return L1Loss(pulse1[1].to(self.device), pulse2[1].to(self.device)).item()

    def _get_info(self): 
        """Return state-related info."""
        info = {
            "current_control": self._observation,
            "distance_from_lower": torch.norm(self.tensor_observation).item(),
            "distance_from_upper": torch.norm(+1 * torch.ones(3) - self.tensor_observation).item(),
            "L1Loss": self._get_control_loss()
        }
        return info

    def reset(self, seed:int=None, options=None)->None: 
        """Resets the environment to initial observations"""
        self._observation = torch.clip(self.rho_zero.sample(), torch.zeros(self.StateDim), torch.ones(self.StateDim))
        self.n_steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs()

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
        healthy_reward = .2  # small constant, reward for having not failed yet.
        loss_penalty = self._get_control_loss()/self.MAX_LOSS  # 0-1 range, in absolute value
        action_penalty = (torch.norm(torch.from_numpy(action)) / torch.sqrt(torch.tensor(3))).item()  # 0-1 range as well

        coeff_healthy, coeff_loss, coeff_drastic = 0.01, 0.99, 0.0  # v1 does not take into account actions magnitude
        return coeff_healthy * healthy_reward - coeff_loss * loss_penalty - coeff_drastic * action_penalty

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
    
    def _render_pulse(self)->np.array: 
        """Renders pulse shape against target.
        
        Returns:
            np.array: PIL image as rgb array. 
        """
        # retrieving control and target pulse and time axis
        _, time, control_pulse = self.laser.forward_pass(self.control_utils.\
                                                         remagnify_descale(self._observation))
        
        # using rendering functions to show off pulses
        fig, ax = visualize_pulses([time, control_pulse], [self.target_time, self.target_pulse])
        # specializing the plots for showcasing trajectories
        ax.legend(loc="upper left", fontsize=12)
        title_string = f"Timestep {self.n_steps}/{self.MAX_STEPS}"
        title_string = title_string if self.n_steps != 0 else "START" + title_string
        ax.set_title(title_string, fontsize=12)

        # creating and coloring the canvas
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        pulses_rgb_array = np.array(Image.fromarray(X).convert('RGB'))
        plt.close(fig)

        return pulses_rgb_array

    def _render_controls(self)->np.array:
        """
        Renders the evolution of control parameters in feasible space with respect to a size `n` deque.
        """
        raise NotImplementedError("This method has not been implemented yet (be patient!).")

    def _render_frame(self): 
        """
        Renders one frame only using Pygame.
        """
        if self.render_mode == "rgb_array":  # returns the transposed rgb array
            return np.transpose(self._render_pulse(), axes=(1, 0, 2))
        
        elif self.render_mode == "human":  # renders using pygame
            screen_size = (640, 480)

            if getattr(self, "window", None) is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(screen_size)  # usual screen size

            if getattr(self, "clock", None) is None:
                self.clock = pygame.time.Clock()
            
            # converting RGB array to something meaningful
            visual_rgb_array = np.transpose(self._render_pulse(), axes=(1, 0, 2))
            # creating the surface from the pulse surface
            pulses_surf = pygame.surfarray.make_surface(visual_rgb_array) 
            # rescaling the surface to fit screen size
            pulses_surf = pygame.transform.scale(pulses_surf, screen_size)
            self.window.blit(pulses_surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

    def close(self):
        if getattr(self, "window", None) is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self, mode="human"):
        """Calls the render frame method."""
        return self._render_frame()