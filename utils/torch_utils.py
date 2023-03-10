import torch
from typing import Iterable
import matplotlib.pyplot as plt

def iterable_to_cuda(input:Iterable[torch.tensor]) -> Iterable[torch.tensor]: 
    """This function returns an iterable containing all the tensors in the input Iterable in which the various tensors
    are sent to CUDA (if applicable). 

    Args:
        input (Iterable[torch.tensor]): Iterable of tensors to be sent to CUDA.

    Returns:
        Iterable[torch.tensor]: Iterable of tensors sent to CUDA. 
    """
    return [tensor.to("cuda") if torch.cuda.is_available() else tensor for tensor in input]

from typing import Tuple

def visualize_pulses(
    pulse:Tuple[torch.tensor, torch.tensor], 
    target_pulse:Tuple[torch.tensor, torch.tensor]
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
    # unpacking inputs
    time, actual_pulse = pulse
    target_time, target_pulse = target_pulse
    
    # retrieving index where time is 0 (not exactly 0, dependings on fft Dt value)
    zero_pos = torch.argwhere(torch.abs(time) == torch.abs(time).min()).item()
    # retrieving peak of pulse
    max_pos = torch.argmax(actual_pulse).item()
    # retrieving peak of target pulse
    target_max_pos = torch.argmax(target_pulse).item()
    # rolling the two pulses to make them centered in 0
    centering_target = -(target_max_pos - zero_pos) if target_max_pos - zero_pos >= 0 else zero_pos - target_max_pos
    # always centering the pulse on zero
    rolled_pulse = torch.roll(
            actual_pulse, 
            shifts = centering_target
            )
    
    fig, ax = plt.subplots()
    # plotting
    ax.plot(
        time.numpy(), 
        rolled_pulse.numpy(), 
        lw = 2, 
        label = "Actual Pulse")

    ax.scatter(
        time.numpy(), 
        torch.roll(target_pulse, 
                shifts = centering_target).numpy(),
        label = "Target Pulse", 
        c = "tab:grey",
        marker = "x", 
        s = 50)
    
    ax.set_xlim(-8e-12, 8e-12)