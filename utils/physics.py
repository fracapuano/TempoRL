import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.fft import fft, fftfreq, fftshift

class Laser: 
    def __init__(self, frequency:np.array, intensity:np.array, syntetic_points:int=int(1e3))->None: 
        self.frequency = frequency
        self.intensity = intensity
    
    def augmentation(self, zero_threshold:float=3e-2, num_points:int=int(1e3), overwrite:bool=True)->None: 
        """This function defines (or re-defines) class attributes related to frequency and intensity as augmented data 
        (using standard interpolation).

        Args:
            zero_threshold (float, optional): Threshold used to define a single element to be zero. Defaults to 3e-2.
            num_points (int, optional): Number of syntetic data points to generate. Defaults to int(1e3).
            overwrite (bool, optional): Whether or not to overwrite the already-defined frequency and intensity values. Defaults to True. 
        """
        # cutting out the frequencies for which intensity is numerically zero (noise included)
        self.start_freq, self.end_freq = sorted((
            self.frequency[np.argwhere(self.intensity.reshape(-1,) > 0 + zero_threshold).reshape(-1,)[0]], 
            self.frequency[np.argwhere(self.intensity.reshape(-1,) > 0 + zero_threshold).reshape(-1,)[-1]])
        )

        self.interpolator = interp1d(self.frequency, self.intensity)

        aug_frequency = sorted(np.linspace(self.start_freq, self.end_freq, num_points, endpoint=True).reshape(-1,))
        aug_intensity = self.interpolator(aug_frequency)

        if overwrite: 
            self.frequency = aug_frequency
            self.intensity = aug_intensity
        else: 
            self.aug_frequency = aug_frequency
            self.aug_intensity = aug_intensity
    
    def wcarrier(self)->None:
        """This function defines the attribute central_carrier (angular central carrier frequency) using the presented data. 
        """ 
        half_max = self.intensity.max()/2
        shifted_intensity = self.intensity - half_max
        
        shifted_model = UnivariateSpline(sorted(self.frequency), shifted_intensity, s=0)
        self.left_freq, self.right_freq = sorted((shifted_model.roots().min(), shifted_model.roots().max()))

        central_carrier = 2 * np.pi * abs(self.left_freq + self.right_freq)/2
        
        self.central_carrier = central_carrier
        return central_carrier
    
    def set_params(self, control_params:np.array=np.array([27.0, 40.0, 50.0]))->None: 
        """This function updates current laser parameters with new ones.

        Args:
            control_params (np.array): The control parameters to be set
        """
        if len(control_params) != 3: 
            raise NotImplementedError("Params different from GDD, TOD and FOD not yet implemented... Please stick to these three for the moment.")

        self.control_params = control_params 

    def update_params(self, new_params:np.array)->None: 
        """This function updates current laser parameters with new ones.

        Args:
            new_params (np.array): New parameters with which to update the current ones.  
        """
        self.control_params = new_params

    def emit_phase(self)->np.array: 
        """This function returns the phase emitted with respect to a considered set of parameters leveraging an analytical expression.

        Args:
            control_params (np.array, optional): Control parameters considered. Defaults to np.array([27.0, 40.0, 50.0]).

        Returns:
            np.array: Phase returned according to the analytical model considered for the considered 
        """

        GDD, TOD, FOD = self.control_params
        # parameters to take into account DAZZLER-to-SPIDER distortion
        alpha_GDD, alpha_TOD, alpha_FOD = 25e3, 30e3, 50e3

        # central carrier frequency
        wcarrier = self.wcarrier()

        theorical_phase = (1/2)*(GDD*1000 - alpha_GDD)*1e-30*(2*np.pi*self.frequency - wcarrier)**2 + \
                (1/6)*(TOD*1000-alpha_TOD)*1e-45*(2*np.pi*self.frequency - wcarrier)**3 + \
                (1/24)*(FOD*1000 - alpha_FOD)*1e-60*(2*np.pi*self.frequency - wcarrier)**4
        
        return theorical_phase

    def temporal_profile(self, num_points_increment:int=int(2e4))->np.array:
        """This function returns the temporal profile of the pulse given intensity and phase. 

        Args:
            num_points_increment (int, optional): Increment in the number of sample points to consider for the fft algorithm. Defaults to int(2e4).

        Returns:
            np.array: Time-representation of intensity.
        """
        step = np.diff(self.frequency)[0]
        sample_points = len(self.intensity) + num_points_increment
        time = fftshift(fftfreq(sample_points, d=abs(step)))

        # padding the spectral intensity and phase to increase sample complexity for the fft algorithm
        spectral_intensity = np.pad(self.intensity, (num_points_increment//2, num_points_increment//2), "constant", constant_values = (0,0))
        spectral_phase = np.pad(self.emit_phase(), (num_points_increment//2, num_points_increment//2), "constant", constant_values = (0,0))
        field = fftshift(fft(spectral_intensity * np.exp(1j * spectral_phase)))
        
        field_squaremodulus = np.real(field * np.conj(field)) # only for casting reasons

        return field_squaremodulus/field_squaremodulus.max()
    
    def control_to_pulse(self, control_params:np.array=np.array([27.0, 40.0, 50.0]))->np.array: 
        """This function returns the temporal pulse shape given the control quantities considered
        """
        self.update_params(control_params)
        pulse = self.temporal_profile()
        
        return pulse
    
    def target_pulse(self, num_points_increment:int=int(2e4), custom_target:bool=False)->None: 
        """This function returns the target pulse shape in the temporal domain.

        Args:
            custom_target (bool, optional): Whether or not a custom target pulse shape has to be used. Defaults to False. In this case, the
            shape corresponding to shortest possible pulse is used. 
        """
        if not custom_target: 
            step = np.diff(self.frequency)[0]
            sample_points = len(self.intensity) + num_points_increment
            time = fftshift(fftfreq(sample_points, d=abs(step)))

            # padding the spectral intensity and phase to increase sample complexity for the fft algorithm
            spectral_intensity = np.pad(self.intensity, (num_points_increment//2, num_points_increment//2), "constant", constant_values = (0,0))
            spectral_phase = np.zeros_like(spectral_intensity)
            field = fftshift(fft(spectral_intensity * np.exp(1j * spectral_phase)))
            
            field_squaremodulus = np.real(field * np.conj(field)) # only for casting reasons

            return field_squaremodulus/field_squaremodulus.max()
