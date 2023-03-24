import torch
import matplotlib.pyplot as plt
from .physics import peak_on_peak
from typing import Tuple, List
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
    # ax.set_xlim(-8e-11, 8e-11)
    ax.legend()

    return fig, ax

def visualize_controls(
        controls_buffer:deque
):
    """Renders a precise control buffer."""
    pass
