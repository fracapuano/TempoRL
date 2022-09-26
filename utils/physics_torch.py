import numpy as np
import torch
from typing import Tuple
from scipy.constants import c
from utils.torch_utils import iterable_to_cuda
from utils.physics import *

cuda_available = torch.cuda.is_available()

def translate_control(central_frequency:float, control:torch.tensor, verse:str = "to_gdd")->torch.tensor: 
        """This function translates the control quantities either from Dispersion coefficients (the di's) to GDD, TOD and FOD using a system of linear equations 
        defined for this very scope or the other way around, according to the string "verse".  

        Args:
            central_frequency (float): Central frequency of the spectrum, expressed in Hz.
            control (torch.tensor): Control quanitities (either the di's or delay information). Must be given in SI units.
            verse (str, optional): "to_gdd" to translate control from dispersion coefficients to (GDD, TOD and FOD), solving Ax = b.
            "to_disp" to translate (GDD, TOD and FOD) to dispersion coefficient left-multiplying the control by A. Defaults to "to_gdd". 

        Returns:
            torch.tensor: The control translated according to the verse considered.
        """
         # central wavelength (using c/f = lambda)
        central_wavelength = c / central_frequency

        a11 = (-2 * torch.pi * c)/(central_wavelength ** 2) #; a12 = a13 = 0
        a21 = (4 * torch.pi * c)/(central_wavelength ** 3); a22 = ((2 * torch.pi * c)/(central_wavelength ** 2))**2 # a23 = 0
        a31 = (-12 * torch.pi * c)/(central_wavelength ** 4); a32 = -(24 * (torch.pi * c) ** 2)/(central_wavelength ** 5); a33 = -((2 * torch.pi * c) / (central_wavelength ** 2)) ** 3

        A = torch.tensor([
            [a11, 0, 0], 
            [a21, a22, 0], 
            [a31, a32, a33]
        ], dtype = torch.float64
        )
        # sending to cuda, if applicable
        A, control = iterable_to_cuda(input = [A, control])

        if verse.lower() == "to_gdd": 
            d2, d3, d4 = control
            # solving the conversion system using forward substitution
            GDD = d2 / A[1,1]; TOD = (d3 - A[2,1] * GDD)/(A[2,2]); FOD = (d4 - A[3,1] * GDD - A[3,2] * TOD)/(A[3,3])
            # grouping the tensors maintaing information on the gradient
            return torch.stack([GDD, TOD, FOD])

        elif verse.lower() == "to_disp": 
            return A @ control if cuda_available else (A @ control).cpu()
        else: 
            raise ValueError('Control translatin is either "to_gdd" or "to_disp"!')

def phase_equation(frequency:torch.tensor, central_frequency:float, control:torch.tensor) -> torch.tensor: 
    """This function returns the phase with respect to the frequency and some control parameters.

    Args:
        frequency (torch.tensor): Tensor of frequencies considered (measured in Hz)
        central_frequency (float): Central frequency, not angular (measured in Hz).
        control (torch.tensor): Control parameters to be used to create the phase. It contains GDD, TOD and FOD in s^2, s^3 and s^4.

    Returns:
        torch.tensor: The phase with respect to the frequency, measured in radiants.
    """
    central_frequency = torch.tensor(central_frequency, dtype = torch.float64)
    frequency, control, central_frequency = iterable_to_cuda(input = [frequency, control, central_frequency])

    GDD, TOD, FOD = control
    phase = \
            (1/2)* GDD * (2*torch.pi * (frequency - central_frequency))**2 + \
            (1/6)* TOD * (2*torch.pi * (frequency - central_frequency))**3 + \
            (1/24)* FOD * (2*torch.pi * (frequency - central_frequency))**4
    
    return phase if torch.cuda.is_available() else phase.cpu()

def yb_gain(signal:torch.tensor, intensity_yb:torch.tensor, n_passes:int=50)->torch.tensor: 
    """This function models the passage of the signal in the cristal in which yb:yab gain is observed.
    
    Args: 
        signal (torch.tensor): The intensity signal that enters the system considered.
        intensity_yb (torch.tensor): The gain intensity of the crystal
        n_passes (int, optional): The number of times the beam passes through the crystal where spectrum narrowing is observed. 
        
    Returns: 
        torch.tensor: New spectrum, narrower because of the gain. 
    """
    n_passes = torch.tensor(n_passes, dtype = torch.float64)
    signal, intensity_yb, n_passes = iterable_to_cuda(input = [signal, intensity_yb, n_passes])
    
    return signal * (intensity_yb ** n_passes)

def impose_phase(spectrum:torch.tensor, phase:torch.tensor)->torch.tensor: 
    """This function imposes a phase on a particular signal.
    
    Args: 
        spectrum (torch.tensor): Tensor representing the signal considered.
        phase (torch.tensor): The phase to impose on the signal.
    
    Returns: 
        torch.tensor: New spectrum with modified phase
    """
    spectrum, phase = iterable_to_cuda(input = [spectrum, phase])
    return spectrum * torch.exp(1j * phase)

def temporal_profile(frequency:torch.tensor, field:torch.tensor, npoints_pad:int=int(1e4), return_time:bool=True) -> Tuple[np.array, np.array]:
    """This function returns the temporal profile of a given signal represented in the frequency domain. Padding is added so as to have more points and increase FFT algorithm output's quality. 

    Args:
        frequency (torch.tensor): Array of frequencies considered (measured in Hz)
        field (torch.tensor): Array of field measured in the frequency domain. 
        npoints_pad (int, optional): Number of points to be used in padding. Padding will be applied using half of this value on the
        right and half on the left. Defaults to int(1e4).
        return_time (bool, optional): Whether or not to return also the time frame of the signal to be used on the x-axis. Defaults to True. 
    Returns:
        Tuple[np.array, np.array]: Returns either (time, intensity) (with time measured in in seconds) or intensity only.
    """
    # send iterable to cuda
    frequency, field = iterable_to_cuda(input = [frequency, field])
    # create the time array
    step = torch.diff(frequency)[0]
    Dt = 1 / step
    time = torch.linspace(start = - Dt/2, end = Dt/2, steps = len(frequency) + npoints_pad)
    # centering the array in its peak - padding the signal extremities to increase resolution
    field = torch.nn.functional.pad(input = field, pad = (npoints_pad//2, npoints_pad//2), mode = "constant", value = 0)
    # going from frequency to time
    field_time = torch.fft.ifftshift(torch.fft.ifft(field))
    # obtaining intensity
    intensity_time = torch.real(field_time * torch.conj(field_time)) # only for casting reasons
    # normalizing the resulting signal
    intensity_time = intensity_time / intensity_time.max() # normalizing
    
    # either returning time or not according to return_time
    if not return_time: 
        return intensity_time if cuda_available else intensity_time.cpu()
    else: 
        return time, intensity_time