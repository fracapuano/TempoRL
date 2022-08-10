from multiprocessing.sharedctypes import Value
import re
import numpy as np
from scipy.fft import fft, fftfreq, fftshift

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

def theorical_phase(frequency:np.array, central_carrier:float)->np.array:
    """This function returns the theorical phase coming out of a DAZZLER-SPIDER system considering specific input frequency and central carrier. The phase
    is computed according to an expert-provided custom formula. 

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
            GDD, TOD, FOD = control_params 
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
