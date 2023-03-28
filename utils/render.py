import torch
import matplotlib.pyplot as plt
from .physics import peak_on_peak
from typing import List
import numpy as np
from collections import deque

def visualize_pulses(
    pulse:List[torch.TensorType], 
    target_pulse:List[torch.TensorType]
):
    """This function visualizes two different pulses rolling up the two to peak-index
    
    Args: 
        pulse (Tuple[torch.tensor, torch.tensor]): Tuple of tensors. First tensor is pulse time axis, second
                                                   tensor is temporal profile of pulse itself. This pulse will
                                                   be plotted with a solid line.
        target_pulse (Tuple[torch.tensor, torch.tensor]): Tuple of tensors. First tensor is pulse time axis, second
                                                   tensor is temporal profile of a target pulse. This will
                                                   be plotted with a scatter plot.
    """
    # centering and unpacking inputs pulses
    [time, actual_pulse], [target_time, target_pulse] = peak_on_peak(
        pulse, 
        target_pulse
    )
        
    fig, ax = plt.subplots()
    # plotting
    ax.plot(
        time.numpy(), 
        actual_pulse.numpy(), 
        lw = 2, 
        label = "Actual Pulse")

    ax.scatter(
        time.numpy(), 
        target_pulse.numpy(),
        label = "Target Pulse", 
        c = "tab:grey",
        marker = "x", 
        s = 50)
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.set_xlim(-8e-12, 8e-12)
    ax.legend()

    return fig, ax

def visualize_controls(
        controls_buffer:deque
):
    """Renders a series of observation in the control space."""
    controls = np.array(controls_buffer, dtype=object)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    # setting bounds to the plot
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    # labels to axis
    ax.set_xlabel("GDD (a.u.)", fontsize=12)
    ax.set_ylabel("TOD (a.u.)", fontsize=12)
    ax.set_zlabel("FOD (a.u.)", fontsize=12)
    
    line, = ax.plot([], [], [], label = "Controls Applied"); scatt = ax.scatter([],[],[], s = 50, c = "red")
    GDDs, TODs, FODs = controls[:, 0], controls[:, 1], controls[:, 2]
    # drawing a line between controls
    line.set_data(GDDs, TODs); line.set_3d_properties(FODs)
    # scatter plot of applied controls
    scatt._offsets3d = GDDs, TODs, FODs
    ax.legend(loc = "upper right", framealpha = 1., fontsize=12)

    return fig, ax
