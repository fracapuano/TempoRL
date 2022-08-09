import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

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

class Laser: 

    def __init__(self, frequency:np.array, intensity:np.array, cutoff_frequencies:np.array, syntetic_points:int=int(1e3))->None: 
        self.frequency = frequency
        self.intensity = intensity
        self.augmented = False

        # TODO cutting out the frequencies for which intensity is numerically zero (noise included)
        # if detect_cutoff: 
        #     self.start_freq, self.end_freq = sorted((
        #         self.frequency[np.argwhere(self.intensity.reshape(-1,) > 0 + zero_threshold).reshape(-1,)[0]], 
        #         self.frequency[np.argwhere(self.intensity.reshape(-1,) > 0 + zero_threshold).reshape(-1,)[-1]])
        #     )
        self.start_freq, self.end_freq = cutoff_frequencies
    
    def augmentation(
        self,
        zero_threshold:float=3e-2, 
        num_points:int=int(1e3), 
        overwrite:bool=True)->None: 
        """This function defines (or re-defines, depending on the value of overwrite) class attributes related to frequency and 
        intensity as augmented data (using standard interpolation).

        Args:
            zero_threshold (float, optional): Threshold used to define a single element to be zero. Defaults to 3e-2.
            num_points (int, optional): Number of syntetic data points to generate. Defaults to int(1e3).
            overwrite (bool, optional): Whether or not to overwrite the already-defined frequency and intensity values. Defaults to True. 

        """            
        self.interpolator = interp1d(self.frequency, self.intensity)

        aug_frequency = np.linspace(self.start_freq, self.end_freq, num_points, endpoint=True)
        aug_intensity = self.interpolator(aug_frequency)

        if overwrite: 
            self.frequency = aug_frequency
            self.intensity = aug_intensity
        else: 
            self.aug_frequency = aug_frequency
            self.aug_intensity = aug_intensity
        
        self.augmented = True
    
    def augment(self)->None: 
        """This method calls the augmentation method if it has not been called so far.
        """
        if not self.augmented: 
            self.augmentation()
    
    def wcarrier(self)->None:
        """This function defines the attribute central_carrier (angular central carrier frequency) using the presented data. 
        """ 
        # check augmentation and, if not, augment
        self.augment()

        half_max = self.intensity.max()/2
        shifted_intensity = self.intensity - half_max

        shifted_model = UnivariateSpline(self.frequency, shifted_intensity, s=0)
        self.left_freq, self.right_freq = shifted_model.roots().min(), shifted_model.roots().max()
    
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
        # check augmentation and, if not, augment
        self.augment()
        
        # parameters to take into account DAZZLER-to-SPIDER distortion
        GDD, TOD, FOD = [27, 40, 50]
        alpha_GDD, alpha_TOD, alpha_FOD = 25e3, 30e3, 50e3

        # central carrier frequency
        wcarrier = self.wcarrier()

        theorical_phase = \
                (1/2)*(GDD*1000 - alpha_GDD)*1e-30*(2*np.pi*self.frequency - wcarrier)**2 + \
                (1/6)*(TOD*1000-alpha_TOD)*1e-45*(2*np.pi*self.frequency - wcarrier)**3 + \
                (1/24)*(FOD*1000 - alpha_FOD)*1e-60*(2*np.pi*self.frequency - wcarrier)**4
        
        return theorical_phase
    
    def phase_expansion(self, custom_phase:np.array=None, degree:int=4)->np.array: 
        """This function polynomially expands a given phase so as to retrieve the control parameters of the final pulse shape.

        Args:
            custom_phase (np.array, optional): Phase shape that can be used. When not provided the output of the "emit_phase" method will be used. Defaults to None.
            degree (int, optional): Degree of the polynomial to be used. Defaults to 4.
        """
        # check augmentation and, if not, augment
        self.augment()

        if custom_phase is None:
            return np.polyfit(self.frequency, self.emit_phase(), deg = degree)[::-1] #from 0-coeff to degree-coeff
        else: 
            return np.polyfit(self.frequency, custom_phase, deg = degree)[::-1]

    def phase_reconstruction(self, control_params:np.array)->np.array:
        """This function uses the control params passed as input to reconstruct the a given phase for given frequencies.

        Args:
            control_params (np.array): Control quantities used to reconstruct the phase.

        Returns:
            np.array: The phase reconstructed with respect to control_params.
        """
        return (self.frequency.reshape(-1,1) ** np.arange(start = 0, stop = len(control_params))) @ control_params

    def temporal_profile(self, control_params:np.array, num_points_increment:int=int(2e4))->np.array:
        """This function returns the temporal profile of the pulse given intensity and phase. 

        Args:
            control_params (np.array): Control parameter to be used in the phase expansion.
            num_points_increment (int, optional): Increment in the number of sample points to consider for the fft algorithm. Defaults to int(2e4).

        Returns:
            np.array: Time-representation of intensity.
        """
        # check augmentation and, if not, augment
        self.augment()

        step = np.diff(self.frequency)[0]
        sample_points = len(self.intensity) + num_points_increment
        time = fftshift(fftfreq(sample_points, d=abs(step)))

        # padding the spectral intensity and phase to increase sample complexity for the fft algorithm
        spectral_intensity = np.pad(self.intensity, (num_points_increment//2, num_points_increment//2), "constant", constant_values = (0,0))
        spectral_phase = np.pad(self.phase_reconstruction(control_params), (num_points_increment//2, num_points_increment//2), "constant", constant_values = (0,0))

        field = fftshift(fft(spectral_intensity * np.exp(1j * spectral_phase)))
        
        field_squaremodulus = np.real(field * np.conj(field)) # only for casting reasons

        return field_squaremodulus/field_squaremodulus.max()
    
    def control_to_pulse(self, control_params:np.array)->np.array: 
        """This function returns the temporal pulse shape given the control quantities considered
        """
        pulse = self.temporal_profile(control_params)
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
        else: 
            raise NotImplementedError("Custom target not yet implemented. Stick to shortest possible pulse for the moment please.")
    
    def laser_loss(self, control_params:np.array, control_bounds:np.array = None, control_penalty:float = 1)->float: 
        """This function returns the scalar (mse) loss related to the pulse obtained with the considered control parameters with respect to the target pulse considered. 

        Args:
            control_params (np.array, optional): Control parameters considered.
            control_bounds (np.array, optional): Bounds to be used to constraint the problem to a considered region (not yet done). Defaults to None.
            control_penalty (float, optional): Penalty term to be used in the objective function to constraint the solution to have one single value. Defaults to 1.

        Returns:
            float: MSE loss value. 
        """
        control_pulse = self.control_to_pulse(control_params=control_params)
        target_pulse = self.target_pulse()

        peak, peak_target = np.argmax(control_pulse), np.argmax(target_pulse)
        delta = peak_target - peak

        return mse(np.roll(control_pulse, delta), target_pulse)