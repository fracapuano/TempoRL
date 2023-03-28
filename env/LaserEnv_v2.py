import torch
import numpy as np
from typing import Tuple, List
from gym.spaces import Box
from torch.distributions.multivariate_normal import MultivariateNormal

from .BaseLaser import Abstract_BaseLaser
from .env_utils import ControlUtils
from utils import physics
from utils.render import visualize_pulses, visualize_controls
from collections import deque
import pygame

import matplotlib.pyplot as plt
from PIL import Image

cuda_available = False
device = "cuda" if cuda_available else "cpu"

class LaserEnv_v2(Abstract_BaseLaser):
    metadata = {
        "render_fps":5, 
        "render_modes": ["rbg_array", "human"]
        }
    
    """Instances a physics-informed version of the L1 Pump laser.
    
    In this version, observations are actual control parameters (that is, \psi). 
    Actions are deltas on said configurations (changes made on \psi), bounded for machine safety.
    The reward depends on the intensity of the controlled pulse. Using an alive bonus, we hope to
    incentivize the agent to stay active.
    Episode termination is declared once the pulse's FWHM goes over a predefinite threshold. 
    """
    def __init__(
    self,
    bounds:torch.TensorType,
    compressor_params:torch.TensorType,
    B_integral:float,
    render_mode:str="rgb_array",
    init_variance:float=.1,
    action_bounds:Tuple[float, List[float]]=0.1,
    device:str=device,
    env_kwargs:dict={})->None:
        """
        Init function. Here laser-oriented and RL-oriented characteristics are defined.
        """
        # device on which to run computation
        self.device = device
        
        # state and action space dimensionality
        self.StateDim = 3
        self.ActionDim = 3

        # custom bounds for env
        if isinstance(action_bounds, list): 
            self.action_lower_bound, self.action_upper_bound = action_bounds
        else:
            self.action_lower_bound, self.action_upper_bound = -action_bounds, +action_bounds
        
        # actual action range
        self.action_range = self.action_upper_bound - self.action_lower_bound

        # keeping a buffer of observations
        self.obs_buffer_size = 5
        self.past_observation_buffer = deque(maxlen=self.obs_buffer_size)

        # reward coefficients
        self.coeffs = env_kwargs.get("reward_coeffs", [1,1])
        # whether or not to reward loss decrements or actual loss values
        self.incremental_improvement = env_kwargs.get("incremental_improvement", False)
        
        # env parametrization init - chagepoint for different xi's.
        super().__init__(
             bounds=bounds, 
             compressor_params=compressor_params, 
             B_integral=B_integral, 
             render_mode=render_mode
        )
        # laser characteristic specifics
        self.transform_limited = self.laser.transform_limited()
        # control utils - suite to handle with ease normalization of control params
        self.control_utils = ControlUtils()  # initialized with default parameters 

        # specifiying obs space
        self.observation_space = Box(
             low = np.zeros(self.StateDim, dtype=np.float32), 
             high = np.ones(self.StateDim, dtype=np.float32)
        )
        
        # actions are defined as deltas - updates are bounded
        self.action_space = Box(
            low = -1 * np.ones(self.ActionDim, dtype=np.float32), 
            high= +1 * np.ones(self.ActionDim, dtype=np.float32)
        )
                
        # defining maximal number of steps and duration (ps) for the pulse
        self.MAX_DURATION = env_kwargs.get("max_duration", 20)
        self.MAX_STEPS = 50
        # physically, the Transform-Limited (TL) pulse maximizes the intensity
        self.TL_intensity = physics.peak_intensity(pulse_intensity=self.transform_limited[1])

        # whether to directly reward the intensity or the increment in intensity
        self.INCREMENTAL = env_kwargs.get("incremental_improvement", False)

        self.rho_zero = MultivariateNormal(
            # loc is compressor params (in the 0-1 range)
            loc=self.control_utils.demagnify_scale(-1 * self.compressor_params).float(), 
            # homoscedastic distribution
            covariance_matrix=torch.diag(init_variance * torch.ones(self.StateDim))
        )
        # setting the simulator in empty state
        self.reset()
    
    @property
    def _observation_SI(self):
        """
        Returns observation in SI units. 
        SI-units observations only are accepted as inputs to the ComputationalLaser considered.
        """
        return self.control_utils.remagnify_descale(self._observation)
    
    @property
    def _pulse(self):
        """Returns the temporal profile of the pulse that derives from the current observation"""
        _, time, control_shape = self.laser.forward_pass(self._observation_SI)
        return (time, control_shape)
    
    @property
    def pulse_FWHM(self):
        """Returns pulse full-width half-maximum. FWHM is given in picoseconds."""
        time, control_shape = self._pulse
        return physics.FWHM(x=time, y=control_shape) * 1e12  # seconds -> picoseconds

    @property
    def peak_intensity(self): 
        """Returns peak intensity of the controlled shape un-doing intensity normalization."""
        return physics.peak_intensity(pulse_intensity=self._pulse[1])
    
    def transform_limited_regret(self): 
        """Computes aligned-L1 loss between current pulse and transform limited"""
        # obtain the pulse shape corresponding to given set of control params
        time, control_shape = self._pulse
        target_time, target_shape = self.transform_limited
        
        # move target and controlled pulse peak on peak
        pulse1, pulse2 = physics.peak_on_peak(
             temporal_profile=[time, control_shape], 
             other=[target_time, target_shape]
             )
        # compute sum(L1 loss)
        return (pulse1[1] - pulse2[1]).abs().sum().item()

    def _get_info(self): 
        """Return state-related info."""
        info = {
            "current_control": self._observation,
            "current_control (SI)": self._observation_SI,
            "current FWHM (ps)": self.pulse_FWHM,
            "current Peak Intensity": self.peak_intensity,
            "L1Loss": self.transform_limited_regret(),
            "DurationStoppage": self.DurationStoppage,
            "TimeStepsStoppage": self.TimeStepsStoppage,
        }
        return info
    
    def remap_action(self, action:np.ndarray)->np.ndarray:
        """
        Remaps the action (sampled from action space) to fit the [self.action_lower_bound,
        self.action_upper bound] range defined for the actual problem.
        For more info on why we need to perform action normalization, consider checking: 
        https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
        Args: 
            action (np.array): Action sampled from self.action_space (in the [-1, +1] range).
        Returns: 
            np.array: Action in the [self.action_lower_bound, self.action_upper_bound] range.
        """
        normalized_range = 2  # normalized_range = normalized_upper - normalized_lower = +1 - (-1) = 2
        ratio = self.action_range / normalized_range
        # re-normalizing (y = lb + ration * (x - (-1)))
        return self.action_lower_bound + ratio * (action + 1)

    def reset(self, seed:int=None, options=None)->None: 
        """Resets the environment to initial observations"""
        # number of steps
        self.n_steps = 0
        # starting in a (feasible) random state
        self._observation = torch.clip(self.rho_zero.sample(), 
                                       torch.zeros(self.StateDim), 
                                       torch.ones(self.StateDim))
        # storing info related to why has the episode stopped
        self.DurationStoppage = False
        self.TimeStepsStoppage = False
        # clearing up the observations buffer
        self.past_observation_buffer.clear()

        return self._get_obs()

    def is_done(self)->bool:
        """
        This function returns a boolean that represents whether or not the optimization process is done
        given the current state.
        """
        reached_maximal_timesteps = self.n_steps >= self.MAX_STEPS
        reached_duration_threshold = self.pulse_FWHM >= self.MAX_DURATION
        
        self.DurationStoppage = reached_duration_threshold
        self.TimeStepsStoppage = reached_maximal_timesteps

        if reached_maximal_timesteps or reached_duration_threshold: 
            return True  # stop episode
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
            float: Value of reward. Sum of two different components. Namely: Alive Bonus and Intensity (either gain or pure value)
        """
        healthy_reward = 2  # small constant, reward for having not failed yet.
        if self.INCREMENTAL: 
            intensity_reward = self.peak_intensity - self.current_intensity  # rewarding variations of intensity
        else: 
            intensity_reward = self.peak_intensity / self.TL_intensity # rewarding intensity itself
        # reward coefficients
        coeff_healthy, coeff_intensity = self.coeffs

        return coeff_healthy * healthy_reward + coeff_intensity * intensity_reward

    def step(self, action:torch.TensorType):
        """
        Applyies given action on laser env. 
        """
        # increment number of steps
        self.n_steps += 1
        # storing the current observation
        self.past_observation_buffer.append(self._observation.numpy())
        # scaling the action to the actual range
        rescaled_action = self.remap_action(action=action)  # here action is in numpy
        # applying (rescaled) action, clipping between 0 and 1
        self._observation = torch.clip(self._observation + torch.from_numpy(rescaled_action), 
                                       min=torch.zeros(3), 
                                       max=torch.ones(3))
        
        reward = self.compute_reward(state=self._observation, action=action)
        done = self.is_done()
        info = self._get_info()
        
        return self._observation, reward, done, info
    
    def _render_pulse(self)->np.array: 
        """Renders pulse shape against target.
        
        Returns:
            np.array: PIL image as rgb array. 
        """
        # retrieving control and target pulse and time axes
        time, control_pulse = self._pulse
        target_time, target_pulse = self.transform_limited
        
        # using rendering functions to show off pulses
        fig, ax = visualize_pulses([time, control_pulse], [target_time, target_pulse])
        
        # specializing the plots for showcasing trajectories
        title_string = f"Timestep {self.n_steps}/{self.MAX_STEPS}"
        if self.n_steps == 0:  # episode start
            title_string = title_string if self.n_steps != 0 else "*** START *** " + title_string
            ax.get_lines()[0].set_color("red")
        
        # text box displays info on current control and transform-limited regret
        knobs = self._observation.tolist()
        control_info = 'GDD: {:2.2e}\n'.format(knobs[0])+\
                       'TOD: {:2.2e}\n'.format(knobs[1])+\
                       'FOD: {:2.2e}\n'.format(knobs[2])+\
                       'B-integral: {:.4f}'.format(self.laser.B)
        
        energy_info = 'L1Loss: {:.4f}\n'.format(self.transform_limited_regret())+\
                      'FWHM (ps): {:2.2f}\n'.format(self._get_info()["current FWHM (ps)"])+\
                      'PI (W/m^2): {:2.2e}'.format(self._get_info()["current Peak Intensity"])
        
        props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)
        ax.text(0.7, 0.95, control_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.text(0.025, 0.8, energy_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

        
        ax.legend(loc="upper left", fontsize=12)
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
        fig, ax = visualize_controls(self.past_observation_buffer)
        # specializing the plots for showcasing trajectories
        title_string = f"Timestep {self.n_steps}/{self.MAX_STEPS}"
        if self.n_steps == 0:  # episode start
            title_string = title_string if self.n_steps != 0 else "*** START *** " + title_string
            ax.get_lines()[0].set_color("red")
        
        ax.set_title(title_string, fontsize=12)

        # creating and coloring the canvas
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        controls_rgb_array = np.array(Image.fromarray(X).convert('RGB'))
        plt.close(fig)

        return controls_rgb_array
    
    def _render_frame(self): 
        """
        Renders one frame only using Pygame.
        """
        if self.render_mode == "rgb_array":  # returns the transposed rgb array
            return np.transpose(self._render_pulse(), axes=(1, 0, 2))
        
        elif self.render_mode == "human":  # renders using pygame
            screen_size = (1280, 480)

            if getattr(self, "window", None) is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(screen_size)  # usual screen size

            if getattr(self, "clock", None) is None:
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            
            # converting RGB array to something meaningful
            pulse_rgb_array = np.transpose(self._render_pulse(), 
                                            axes=(1, 0, 2))
            controls_rgb_array = np.transpose(self._render_controls(),
                                              axes=(1, 0, 2))
            
            # creating the surface from pulse array
            pulses_surf = pygame.surfarray.make_surface(pulse_rgb_array) 
            # creating the surface from the control array
            controls_surf = pygame.surfarray.make_surface(controls_rgb_array) 
            # rescaling the surfaces to fit screen size
            pulses_surf = pygame.transform.scale(pulses_surf, (screen_size[0]//2, screen_size[1]))
            controls_surf = pygame.transform.scale(controls_surf, (screen_size[0]//2, screen_size[1]))
            self.window.blit(pulses_surf, (0, 0))
            self.window.blit(controls_surf, (screen_size[0]//2, 0))
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