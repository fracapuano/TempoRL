import numpy as np
from typing import Tuple
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

def cutoff_signal(frequency_cutoff:Tuple[float, float], frequency:np.array, signal:np.array) -> Tuple[np.array, np.array]:
    """This function cuts the input signal using input frequency cutoff. It returns the cutted signal. The cut is done so as to discard
    from the analysis the part of the signal which is non-zero only for measurement noise.

    Args:
        frequency_cutoff (Tuple[float, float]): Lower and upper cutoff frequency, measured in THz.
        frequency (np.array): Array of frequencies considered, measured in THz.
        signal (np.array): The signal of interest (represented in the frequency).

    Raises:
        ValueError: Lower cutoff frequency must always be strictly smaller than upper cutoff frequency.

    Returns:
        Tuple[np.array, np.array]: Cutted arrays of frequency and field, (frequency, field). 
    """
    low_f, up_f = frequency_cutoff
    if low_f >= up_f: # check on consistency
        raise ValueError("frequency_cutoff must be a tuple of strictly increasing values.")
    left_idx, right_idx = np.argwhere(frequency >= low_f)[0].item(), np.argwhere(frequency <= up_f)[-1].item()
    return frequency[left_idx:right_idx+1], signal[left_idx:right_idx+1]

def equidistant_points(frequency:np.array, signal:np.array, num_points:int=int(5e3)) -> Tuple[np.array, np.array]: 
    """This function uses a spline to interpolate the input signal so as to be able to reproduce it for
    equidistant frequencies rather than for the actual sampled ones (which are not equidistant).

    Args:
        frequency (np.array): Array of frequencies considered measured in THz.
        signal (np.array): The signal of interest (represented in the frequency domain).

    Returns:
        Tuple[np.array, np.array]: Equidistant array of frequencies, Correspondant values of electric field, (frequency, field). 
    """
    spline = UnivariateSpline(sorted(frequency), signal, s = 0) # use all points available
    equidistant_frequency = np.linspace(start = frequency.min(), stop = frequency.max(), num = num_points)

    return (equidistant_frequency, spline(equidistant_frequency))

def central_frequency(frequency:np.array, signal:np.array) -> float: 
    """This function computes the central frequency of a given signal. 
    The central frequency is defined as the average frequency between the two edge frequencies of full width half
    maximum.

    Args:
        frequency (np.array): Array of frequencies, measured in Hz.
        signal (np.array): The signal of interest (represented in the frequency domain).

    Returns:
        float: Central (not angular) frequency. 
    """
    half_max = signal.max() / 2
    spline = UnivariateSpline(frequency, signal - half_max, s = 0) # use all the points available   
    left_freq, right_freq = spline.roots().min(), spline.roots().max()
    # returns central frequency (in Hz, not angular)
    return abs(left_freq + right_freq)/2

def phase_equation(frequency:np.array, central_frequency:float, GDD:float, TOD:float, FOD:float) -> np.array: 
    """This function returns the phase with respect to the frequency and some control parameters. The various multiplicative
    constants present in phase's formula derive mainly from the necessity of converting everything to SI uom.

    Args:
        frequency (np.array): Array of frequencies considered (measured in Hz)
        central_frequency (float): Central frequency, not angular, measured in Hz.
        GDD (float): Group Delay Dispersion, measured in 10^{-30} s^2 (femtoseconds squared).
        TOD (float): Control parameter 1, measured in {10^-45} s^3 (femtoseconds cubed). 
        FOD (float): Control parameter 1, measured in {10^-60} s^4 (femtoseconds to the fourth). 

    Returns:
        np.array: The phase with respect to the frequency, measured in radiants.
    """
    phase = \
            (1/2)* GDD * 1e-30 * (2*np.pi * (frequency - central_frequency))**2 + \
            (1/6)* TOD * 1e-45 * (2*np.pi * (frequency - central_frequency))**3 + \
            (1/24)* FOD * 1e-60 * (2*np.pi * (frequency - central_frequency))**4
    return phase

def mse(x:np.array, y:np.array)->float: 
    """This function computes the MSE between x and y.

    Args:
        x (np.array): First input array
        y (np.array): Second input array

    Returns:
        float: MSE value.
    """
    x, y = x.reshape(-1,), y.reshape(-1,)
    return ((x-y)**2).mean()