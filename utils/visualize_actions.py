import matplotlib.pyplot as plt
import numpy as np
import torch
from .physics import PulseEmbedding
from matplotlib.animation import FuncAnimation, FFMpegWriter

class AnimateActions: 
    def __init__(self, performed_actions:np.array, laser_wrapper:object, last_actions:int=5) -> None: 
        """Init function.

        Args: 
            performed_actions (np.array): ndarray containing all the actions performed in a given optimisation
            route. Shape of (n_points, actions).
            laser (object): laser model to be used to forward the actions and observe pulse shape.
            last_actions (int, optional): Number of actions to show in control visualizations.

        Raises: 
            ValueError: Performed actions must be a np.ndarray. 
        """
        if not isinstance(performed_actions, np.ndarray): 
            raise ValueError("Array containing performed actions must be a numpy ndarray of shape (n_actions, action_dimensionality)")
        
        self.actions = performed_actions
        self.laser_wrapper = laser_wrapper
        self.laser = self.laser_wrapper.LaserModel
        self.embedder = self.laser_wrapper.PulseEmbedder
        self.last_actions = last_actions
        
    def control_animation(self, i:int):
        """This function defines the animation function to be called at every step of the animation itself.

        Args:
            i (int): Iterator index. Used by the animator 
        """
        self.ax.set_title(f"Iteration {i+1}/{len(self.actions)}")
        if i<self.last_actions:
            GDDs, TODs, FODs = self.actions[:i, 0], self.actions[:i, 1], self.actions[:i, 2]
            self.line.set_data(GDDs, TODs); self.line.set_3d_properties(FODs)
            self.scatt._offsets3d = GDDs, TODs, FODs
        else:
            GDDs, TODs, FODs = self.actions[i-self.last_actions:i, 0], self.actions[i-self.last_actions:i, 1], self.actions[i-self.last_actions:i, 2]
            self.line.set_data(GDDs, TODs); self.line.set_3d_properties(FODs)
            self.scatt._offsets3d = GDDs, TODs, FODs
        return self.line, 
        
    def animate_controls(self, interval:float=10, save_anim:bool=False, fname:str="control_anim.mp4")->None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection = "3d")
        # setting bounds to the plot
        self.ax.set_xlim3d(self.actions[:,0].min(), self.actions[:,0].max())
        self.ax.set_ylim3d(self.actions[:,1].min(), self.actions[:,1].max());
        self.ax.set_zlim3d(self.actions[:,2].min(), self.actions[:,2].max());
        # labels to axis
        self.ax.set_xlabel(r"GDD ($s^2$)"); self.ax.set_ylabel(r"TOD ($s^3$)"); self.ax.set_zlabel(r"FOD ($s^4$)"); 

        self.line, = self.ax.plot([], [], [], label = "Optimization Route"); self.scatt = self.ax.scatter([],[],[], s = 50, c = "red")
        self.ax.scatter(*-1*self.laser.compressor_params, c = "red", marker = "*", s = 100, label = r"$\varphi_{str.} = -\varphi_{compr.}$")

        # run animation
        anim = FuncAnimation(self.fig, self.control_animation, frames = len(self.actions), interval = interval, repeat = False)
        self.ax.legend(loc = "upper right", framealpha = 1.)

        if save_anim: 
            writervideo = FFMpegWriter(fps = 10)
            anim.save(fname, writer = writervideo)

    def pulse_animation(self, i:int)->None:
        """This function defines the animation function to be called at every step of the animation itself.

        Args:
            i (int): Iterator index. Used by the animator 
        """
        # initialize an empty ax
        self.ax_pulse = self.ax
        
        # obtain transform limited
        _, target_pulse = self.laser.transform_limited()
        # obtain time and pulse with respect to given actions
        time, pulse = self.laser.forward_pass(control = torch.from_numpy(self.actions[i])) 
        if torch.cuda.is_available(): 
            time = time.cpu()
            pulse = pulse.cpu()
        
        # clear previous frame
        self.ax_pulse.cla()
        # the position of the zero will always be the median position for time
        zero_pos = torch.argwhere(torch.abs(time) == torch.abs(time).min()).item()
        max_pos = torch.argmax(pulse).item()
        centering = -(max_pos - zero_pos) if max_pos - zero_pos >= 0 else zero_pos - max_pos

        target_max_pos = torch.argmax(target_pulse).item()
        centering_target = -(target_max_pos - zero_pos) if target_max_pos - zero_pos >= 0 else zero_pos - target_max_pos
        
        # always centering the pulse on zero
        rolled_pulse = torch.roll(
            pulse, 
            shifts = centering
            )
        
        # (slightly augmented) buildup duration as xlim
        embed_series = self.embedder.embed_Series(time = time.numpy(), pulse = rolled_pulse.numpy())    

        self.ax_pulse.plot(time.numpy(), rolled_pulse.numpy(), lw = 2, label = f"Iteration {i+1}/{len(self.actions)}")

        if self.show_target:
            self.ax_pulse.scatter(
                time.numpy(), 
                np.roll(target_pulse.numpy(), shift = centering_target),
                label = "Target Pulse", 
                c = "tab:grey",
                marker = "x", 
                s = 50)

        # slightly bigger than the buildup
        left_lim = -8e-10 if embed_series["Buildup_left"] <= -8e-10 else -8e-12
        right_lim = 8e-10 if embed_series["Buildup_right"] >= 8e-10 else +8e-12
        embed_series["Buildup_right"]
        self.ax_pulse.set_xlim(left = left_lim, right = right_lim)
        
        self.ax_pulse.legend(fontsize = 12, loc = "upper right", framealpha = 1.)
    
    def animate_actions(self, interval:float=10, show_target:bool=True, save_anim:bool=False, fname:str="pulses_anim.mp4")->None:
        """This function animates the performed actions
        """ 
        self.fig, self.ax = plt.subplots()
        self.show_target = show_target
        # run animation
        anim = FuncAnimation(self.fig, self.pulse_animation, frames = len(self.actions), interval = interval, repeat = False)
        if save_anim: 
            writervideo = FFMpegWriter(fps = 10)
            anim.save(fname, writer = writervideo)