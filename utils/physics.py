import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift
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
    step = np.diff(frequency)[0]
    time = fftshift(fftfreq(len(frequency) + npoints_pad, d=abs(step)))

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

# L3 necessary information - retrieved from previous version - TO BE INTEGRATED SOON
def theorical_phase(frequency:np.array, central_carrier:float)->np.array:
    """This function returns the theorical phase coming out of a DAZZLER-SPIDER system considering specific input frequency and central carrier. The phase
    is computed according to an expert-provided custom formula. The uoms of GDD, TOD and FOD are not SI units but are fs^2, fs^3 and fs^3. 

    Args:
        frequency (np.array): Array of frequencies considered. 
        central_carrier (float): Central carrier frequency. Used to emit the phase.

    Returns:
        np.array: Theorical phase obtained considering the formula presented.
    """
    GDD, TOD, FOD = 30, 40, 50
    alpha_GDD, alpha_TOD, alpha_FOD = 25e3, 30e3, 50e3

    theorical_phase = \
            (1/2)*(GDD*1000 - alpha_GDD)*1e-30*(2*np.pi*frequency - central_carrier)**2 + \
            (1/6)*(TOD*1000-alpha_TOD)*1e-45*(2*np.pi*frequency - central_carrier)**3 + \
            (1/24)*(FOD*1000 - alpha_FOD)*1e-60*(2*np.pi*frequency - central_carrier)**4
    
    return theorical_phase

def phase_expansions(frequency:np.array, phase:np.array, degree:int=4)->np.array:
    """This function polynomially expand a phase with respect to input frequency and returns the polynomial coefficients in ascendind order (ordered by degree)

    Args:
        frequency (np.array): Array of frequencies considered. 
        phase (np.array): Array containing the phase to be used in the polynomial fitting process.
        degree (int, optional): Degree of the polynomial used. 

    Returns:
        np.array: The polynomial coefficients used to fit the polynomial. 
    """
    control_params = np.polyfit(frequency, phase, deg = degree)
    return control_params[::-1]

class PulseEmitter: 
    def __init__(self, frequency:np.array, field:np.array, useEquation:bool=False, central_carrier:float=None)->None: 
        self.frequency = frequency 
        self.field = field
        self.useEquation = useEquation
        
        if self.useEquation: 
            if central_carrier is None: 
                raise ValueError("Central carrier needed if the phase is being reconstructed using an equation")
            else: 
                self.central_carrier = central_carrier
    
    def phase_reconstruction(self, control_params:np.array)->np.array: 
        return (self.frequency.reshape(-1,1) ** np.arange(start = 0, stop = len(control_params))) @ control_params

    def phase_control(self, control_params:np.array)->np.array: 
        if len(control_params)==3:
            GDD, TOD, FOD = control_params # the control happens with parameters in fs^2, fs^4 and fs^4.
            alpha_GDD, alpha_TOD, alpha_FOD = 25e3, 30e3, 50e3
        else: 
            raise ValueError("Please use GDD, TOD and FOD only if constructing the phase starting from equation.")
        
        theorical_phase = \
            (1/2)*(GDD*1000 - alpha_GDD)*1e-30*(2*np.pi*self.frequency - self.central_carrier)**2 + \
            (1/6)*(TOD*1000-alpha_TOD)*1e-45*(2*np.pi*self.frequency - self.central_carrier)**3 + \
            (1/24)*(FOD*1000 - alpha_FOD)*1e-60*(2*np.pi*self.frequency - self.central_carrier)**4
    
        return theorical_phase

    def temporal_profile(self, control_params:np.array, num_points:int=int(2e4))->np.array: 
        """This function returns the temporal profile of a signal described in frequency and electric field given the considered control parameters. 
        It applies padding to the arrays to increase the precision of the fft algorithm (since it increases the sample complexity).

        Args:
            control_params (np.array): The coefficients to reconstruct the phase according to a polynomial model.
            num_points (int, optional): Number of points to be used in padding to increase sample complexity. Defaults to int(2e4).

        Returns:
            np.array: The time representation of the pulse. 
        """
        self.npoints_increment = num_points

        spectrum_field = np.pad(self.field, (num_points//2, num_points//2), "constant", constant_values = (0,0))
        if self.useEquation: 
            spec_phase = self.phase_control(control_params)
        else: 
            spec_phase = self.phase_reconstruction(control_params)
            
        spectrum_phase = np.pad(spec_phase, (num_points//2, num_points//2), "constant", constant_values = (0,0))
        
        time_field = fftshift(fft(spectrum_field*np.exp(1j*spectrum_phase)))

        pulse = time_field * np.conjugate(time_field)

        return np.real(pulse / pulse.max())

    def time_scale(self)->np.array:
        """This function returns the time scale for the considered signal.

        Returns:
            np.array: Time scale of the signal (in femtoseconds)
        """
        step = np.diff(self.frequency)[0]
        sample_points = len(self.field) + self.npoints_increment
        time = fftshift(fftfreq(sample_points, d=abs(step)))

        return time * 1e15