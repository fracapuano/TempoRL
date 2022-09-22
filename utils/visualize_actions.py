import matplotlib.pyplot as plt
import numpy as np
import torch
from .physics import PulseEmbedding
from matplotlib.animation import FuncAnimation

class AnimateActions: 
    def __init__(self, performed_actions:np.array, laser:object) -> None: 
        """Init function.

        Args: 
            performed_actions (np.array): ndarray containing all the actions performed in a given optimisation
            route. 
            laser (object): laser model to be used to forward the actions and observe pulse shape.
        
        Raises: 
            ValueError: Performed actions must be a np.ndarray. 
        """
        if not isinstance(performed_actions, np.ndarray): 
            raise ValueError("Array containing performed actions must be a numpy ndarray of shape (n_actions, action_dimensionality)")
        
        self.actions = performed_actions
        self.laser = laser
        self.embedder = PulseEmbedding()
        
        # initialize an empty figure
        fig, ax = plt.subplots()
        self.fig, self.ax = fig, ax 

    def pulse_animation(self, i:int)->None:
        """This function defines the animation function to be called at every step of the animation itself.

        Args:
            i (int): Iterator index. Used by the animator 
        """
        # obtain time and pulse with respect to given actions
        time, pulse = self.laser.forward_pass(control = torch.from_numpy(self.actions[i])) 
        if torch.cuda.is_available(): 
            time = time.cpu()
            pulse = pulse.cpu()
        
        # (slightly augmented) buildup duration as xlim
        embed_series = self.embedder.embed_Series(time = time.numpy(), pulse = pulse.numpy())

        # delete previous frame
        self.ax.cla()

        self.ax.plot(time.numpy(), pulse.numpy(), lw = 2)
        self.ax.set_title(f"Action {i+1}/{len(self.actions)}")

        self.ax.set_xlim(left = 0.9 * embed_series["Buildup_left"], right = 1.1 * embed_series["Buildup_right"])
        
    def animate_actions(self, interval:float=10)->None:
        """This function animates the performed actiosn
        """ 
        # run animation
        anim = FuncAnimation(self.fig, self.pulse_animation, frames = len(self.actions), interval = interval, repeat = False)
        plt.show()

