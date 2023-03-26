import numpy as np
import torch
from scipy.constants import c

from .torch_utils import iterable_to_cuda
from scipy.interpolate import UnivariateSpline

from typing import Tuple, Union, List

cuda_available = False


def translate_control(central_frequency:float, control:torch.TensorType, verse:str = "to_gdd")->torch.TensorType: 
        """This function translates the control quantities either from Dispersion coefficients (the di's) to GDD, TOD and FOD using a system of 
        linear equations defined for this very scope or the other way around, according to the "verse" argument.  

        Args:
            central_frequency (float): Central frequency of the spectrum, expressed in Hz.
            control (torch.tensor): Control quanitities (either the di's or delay information). Must be given in SI units.
            verse (str, optional): "to_gdd" to translate control from dispersion coefficients to (GDD, TOD and FOD), solving Ax = b.
                                    "to_disp" to translate (GDD, TOD and FOD) to dispersion coefficient left-multiplying the control by A. 
                                    Defaults to "to_gdd". 

        Returns:
            torch.tensor: The control translated according to the verse considered.
        """
         # central wavelength (using c/f = lambda)
        central_wavelength = c / central_frequency

        a11 = (-2 * torch.pi * c)/(central_wavelength ** 2) #; a12 = a13 = 0
        a21 = (4 * torch.pi * c)/(central_wavelength ** 3); a22 = ((2 * torch.pi * c)/(central_wavelength ** 2))**2 # a23 = 0

        a31 = (-12 * torch.pi * c)/(central_wavelength ** 4)
        a32 = -(24 * (torch.pi * c) ** 2)/(central_wavelength ** 5)
        a33 = -((2 * torch.pi * c) / (central_wavelength ** 2)) ** 3

        # conversion matrix
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
            raise ValueError('Control translation is either "to_gdd" or "to_disp"!')

def phase_equation(frequency:torch.TensorType, central_frequency:float, control:torch.TensorType) -> torch.tensor: 
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
    
    return phase if cuda_available else phase.cpu()

def yb_gain(signal:torch.TensorType, intensity_yb:torch.TensorType, n_passes:int=50)->torch.TensorType: 
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

def impose_phase(spectrum:torch.TensorType, phase:torch.TensorType)->torch.TensorType: 
    """This function imposes a phase on a particular signal.
    
    Args: 
        spectrum (torch.tensor): Tensor representing the signal considered.
        phase (torch.tensor): The phase to impose on the signal.
    
    Returns: 
        torch.tensor: New spectrum with modified phase
    """
    spectrum, phase = iterable_to_cuda(input = [spectrum, phase])
    return spectrum * torch.exp(1j * phase)

def temporal_profile(frequency:torch.TensorType, field:torch.TensorType, npoints_pad:int=int(1e4), return_time:bool=True) -> Tuple[torch.tensor, torch.tensor]:
    """This function returns the temporal profile of a given signal represented in the frequency domain. Padding is added so as to have more points and increase FFT algorithm 
    output's quality. 

    Args:
        frequency (torch.tensor): Array of frequencies considered (measured in Hz)
        field (torch.tensor): Array of field measured in the frequency domain. 
        npoints_pad (int, optional): Number of points to be used in padding. Padding will be applied using half of this value on the
        right and half on the left. Defaults to int(1e4).
        return_time (bool, optional): Whether or not to return also the time frame of the signal to be used on the x-axis. Defaults to True. 
    Returns:
        Tuple[torch.tensor, torch.tensor]: Returns either (time, intensity) (with time measured in in seconds) or intensity only.
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
        return time, intensity_time if cuda_available else time.cpu(), intensity_time.cpu()

def FWHM(x:torch.TensorType, y:torch.TensorType, return_roots:bool=False)->Union[float, Tuple[float, float]]: 
    """This function computes the FWHM roots of a given signal.

    Args:
        x (torch.tensor): x-axis representation of the signal
        y (torch.tensor): y-axis representation of the signal
        return_roots (bool, optional): whether to return or not return the roots of the half-spline

    Returns:
        Union[float, Tuple[float, float]]: value, in seconds, of FWHM and, optionally, the roots of the half spline.
    """
    # casting to numpy
    x, y = x.numpy(), y.numpy()

    half_signal = y - (y.max() / 2)
    half_spline = UnivariateSpline(x = x, y = half_signal, s = 0)
    
    r1, r2 = half_spline.roots()[0], half_spline.roots()[-1]
    FWHM_value = np.abs(np.array((r1, r2))).sum()
    if return_roots: 
        return FWHM_value, (r1, r2)
    else: 
        return FWHM_value

def FWPercM(x:torch.TensorType, y:torch.TensorType, perc:float, return_roots:bool=False)->Union[float, Tuple[float, float]]: 
    """This function computes the FW Percentual Max roots of a given signal.

    Args:
        x (torch.tensor): x-axis representation of the signal
        y (torch.tensor): y-axis representation of the signal
        perc (float): The value of the percentage of the max to retrieve.
        return_roots (bool, optional): whether to return or not return the roots of the half-spline

    Returns:
        Union[float, Tuple[float, float]]: value, in seconds, of FWHM and, optionally, the roots of the half spline.
    """
    x, y = x.numpy(), y.numpy()

    perc_signal = y - (perc * y.max())
    perc_spline = UnivariateSpline(x = x, y = perc_signal, s = 0)
    
    r1, r2 = perc_spline.roots()[0], perc_spline.roots()[-1]
    FWPercM_value = np.abs(np.array((r1, r2))).sum()
    if return_roots: 
        return FWPercM_value, (r1, r2)
    else: 
        return FWPercM_value
    
def buildup(time:torch.TensorType, intensity:torch.TensorType, return_instants=False, min_thresh:float=1e-3) -> Union[float, Tuple[float, float]]: 
    """This function computes the buildup instants for the given signal.

    Args:
        time (torch.tensor): Time Axis for the representation of the signal
        intensity (torch.tensor): Intensity signal depending on time. 
        return_instants (bool, optional): whether to return or not return the first and last instant of the build-up.
        min_thresh (float, optional): Threshold below which each intensity value is equal to 0. 

    Returns:
        Union[float, Tuple[float, float]]: value, in seconds, of build duration and, optionally, the first and last instant 
        in which such condition is verified.
    """
    x, y = x.numpy(), y.numpy()

    above_mask = intensity > min_thresh
    first_instant, last_instant = time[above_mask][0], time[above_mask][-1]
    buildup_duration = np.abs(np.array((first_instant, last_instant))).sum()
    if return_instants: 
        return buildup_duration, (first_instant, last_instant)
    else: 
        return buildup_duration

def peak_intensity(pulse_intensity:torch.TensorType, w0:float=12e-3, E:float=220e-3, dt:float=4.67e-14) -> float: 
    """
    This function computes the peak intensity given a pulse shape in the 0-1 range and parameters of the energy.
    
    Args:  
        pulse_intensity (np.array): array of intensities in a different range from the actual one given the arbitrary units of the intensity values.
        w0 (float, optional): beam radius of the beam (given in SI units), defaults to 12mm for L1 pump. Defaults to 12 mm. 
        E (float, optional): beam energy (given in SI units), defaults to 220 mJ for L1 pump. Defaults to 220 mJ.
        dt (float, optional): distance in time domain between sample points to approximate the intensity integral. Defaults to 4.67e-14 (s) for 33k elements time profile. 
    
    Returns: 
        float: peak intensity value.
    """

    # the integral of the intensity can be approximated from the area under the curve of the pulse.
    I_integral = torch.trapz(y = pulse_intensity, dx = dt).item()
    return (2 * E) / (np.pi * (w0 ** 2) * I_integral)

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
    spline = UnivariateSpline(frequency, signal - half_max, s = 0)  # use all the points available   
    left_freq, right_freq = spline.roots().min(), spline.roots().max()
    # returns central frequency (in Hz, not angular)
    return abs(left_freq + right_freq)/2

def peak_on_peak(temporal_profile:List[torch.TensorType], other:List[torch.TensorType])->List[List[torch.TensorType]]: 
    """This function shifts the two temporal profiles considered to make them centered in the same moment. That is, to have them be peak on peak.
    Args: 
        temporal_profile (List[torch.TensorType]): One temporal profile. First element represents time axis while second element represents the actual
                                                   pulse shape.
        other (List[torch.TensorType]): Other temporal profile (usually, target one). First element represents time axis while second element represents 
                                        the actual pulse shape.
    
    Returns: 
        List[List[torch.TensorType], List[torch.TensorType]: List of lists of peak on peak tensors.
    """
    # unpacking inputs
    time, actual_pulse = temporal_profile
    target_time, target_pulse = other
    
    # retrieving index where time is 0 (not exactly 0, dependings on fft Dt value)
    zero_pos = torch.argwhere(torch.abs(time) == torch.abs(time).min())[0].item()
    # retrieving peak of pulse
    max_pos = torch.argmax(actual_pulse).item()
    # retrieving peak of target pulse
    target_max_pos = torch.argmax(target_pulse).item()
    # rolling the two pulses to make them centered in 0 - target pulse always peaks in 0
    centering_target = -(max_pos - target_max_pos) if max_pos - target_max_pos >= 0 else target_max_pos - max_pos
    # always centering the pulse on zero
    rolled_pulse = torch.roll(
            actual_pulse, 
            shifts = centering_target
            )
    
    return [[target_time, rolled_pulse], [target_time, target_pulse]]
