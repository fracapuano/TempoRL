import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift
from typing import Tuple
from scipy.interpolate import UnivariateSpline
from utils.se import get_project_root
import pandas as pd
from scipy.constants import c

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

def time_from_frequency(frequency:np.array, pad_points: int) -> np.array:
    """This function generates a temporal vector based on the frequency vector and the ammount of zero padded points.

    Args:
        frequency (np.array): Array of frequencies considered measured in THz.
        pad_points (np.array): The ammount of zero padded points.

    Returns:
        np.array: Equidistant array of time 
    """
    step = np.diff(frequency)[0]
    Dt = 1 / step
    time =  np.linspace(start = -Dt/2, stop = +Dt/2, num = pad_points + len(frequency))    
    return time

def amplification(frequency:np.array, field:np.array, n_passes:int=50, num_points:int=int(5e3)) -> np.array: 
        r"""This function reproduces the effect that passing through a non-linear cristal has on the beam itself. In particular, this function applies
        the modification present in data/cristal_gain.txt to the spectrum coming out of the spectrum effectively modifying it. 

        Args:
            frequency (np.array): Array of frequencies considered.
            field (np.array): Array of electric field represented in frequency domain.
            n_passes (int, optional): Number of passes through the non linear cristal.

        Returns:
            np.array: The field in the laser (\tilde{y}_1), which is the result of the spectrum modification carried out in the non linear cristal.
        """
         # reading the data with which to amplify the signal when non specific one is given
        cristal_path = str(get_project_root()) + "/data/cristal_gain.txt"
        gain_df = pd.read_csv(cristal_path, sep = "  ", skiprows = 2, header = None, engine = "python")
        gain_df.columns = ["Wavelength (nm)", "Intensity"]
        gain_df.Intensity = gain_df.Intensity / gain_df.Intensity.values.max()
        gain_df["Frequency (THz)"] = gain_df["Wavelength (nm)"].apply(lambda wl: 1e12 * (c/((wl+1) * 1e-9))) # 1nm shift

        gain_df.sort_values(by = "Frequency (THz)", inplace = True)
        yb_frequency, yb_field = gain_df["Frequency (THz)"].values, np.sqrt(gain_df["Intensity"].values)
        # cutting the gain frequency accordingly
        yb_frequency, yb_field = cutoff_signal(
            frequency_cutoff=(frequency[0], frequency[-1]), 
            frequency = yb_frequency * 1e-12, 
            signal = yb_field)
        
        # augmenting the cutted data
        yb_frequency, yb_field = equidistant_points(
            frequency = yb_frequency, 
            signal = yb_field, 
            num_points = num_points
        )
        
        amp_field = yb_field * field

        for _ in range(1, n_passes): 
            amp_field *=  yb_field
        
        return amp_field


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
            (1/2)* GDD * (2*np.pi * (frequency - central_frequency))**2 + \
            (1/6)* TOD * (2*np.pi * (frequency - central_frequency))**3 + \
            (1/24)* FOD * (2*np.pi * (frequency - central_frequency))**4
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

def temporal_profile(frequency:np.array, field:np.array, phase:np.array, npoints_pad:int=int(1e4), return_time:bool=True) -> Tuple[np.array, np.array]:
    """This function returns the temporal profile of a given signal considering the signal itself (in the frequency domain) and a given phase. 
    Padding is added so as to have more points and increase FFT algorithm output's quality. 

    Args:
        frequency (np.array): Array of frequencies considered (measured in Hz)
        field (np.array): Array of field measured in the frequency domain. 
        phase (np.array): Array representing the phase considered in the frequency domain. 
        npoints_pad (int, optional): Number of points to be used in padding. Padding will be applied using half of this value on the
        right and half on the left. Defaults to int(1e4).
        return_time (bool, optional): Whether or not to return also the time frame of the signal to be used on the x-axis. Defaults to True. 
    Returns:
        Tuple[np.array, np.array]: Returns either (time, intensity) (with time measured in in femtoseconds) or intensity only.
    """
    time = time_from_frequency(frequency, npoints_pad)
    field_padded = np.pad(field, pad_width=(npoints_pad // 2, npoints_pad // 2), mode = "constant", constant_values = (0, 0))
    phase_padded = np.pad(phase, pad_width=(npoints_pad // 2, npoints_pad // 2), mode = "constant", constant_values = (0, 0))

    field_time = fftshift(fft(field_padded * np.exp(1j * phase_padded))) # inverse FFT to go from frequency domain to temporal domain
    intensity_time = np.real(field_time * np.conj(field_time)) # only for casting reasons

    intensity_time = intensity_time / intensity_time.max() # normalizing
    
    # either returning time or not according to return_time
    if not return_time: 
        return intensity_time
    else: 
        return time, intensity_time

def FWHM(x:np.array, y:np.array)->float: 
    """This function computes the FWHM roots of a given signal.
    Args: 
        x (np.array): x-axis representation of the signal
        y (np.array): y-axis representation of the signal
    
    Returns: 
        float: value, in seconds, of FWHM.
    """
    half_signal = y - (y.max() / 2)
    half_spline = UnivariateSpline(x = x, y = half_signal, s = 0)
    
    return (np.abs(half_spline.roots())).sum()