from typing import Iterable, List
from .LaserModel import *

default_compressor_params = -1*torch.tensor((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50), dtype=torch.float64)
control_lower_bound = torch.tensor((2.3522e-22, -1.003635e-34, 4.774465e-50), dtype=torch.float64)
control_upper_bound = torch.tensor((2.99624e-22, 9.55955e-35, 1.4323395e-49), dtype=torch.float64)


def instantiate_laser(
        compressor_params:Iterable[float], 
        B_integral:float
        )->object: 
    """This function instantiates a Laser Model object based on a given parametrization (in terms 
    of B integral and compressor params).

    Returns:
        object: LaserModel-v2 object.
    """
    frequency, field = extract_data()  # extracts the data about input spectrum from data folde
    cutoff = np.array((289.95, 291.91)) * 1e12  # defines cutoff frequencies
    # cutting off the signal
    frequency_clean, field_clean = cutoff_signal(frequency_cutoff = cutoff, 
                                                 frequency = frequency * 1e12,
                                                 signal = field)
    # augmenting the signal (using splines)
    frequency_clean_aug, field_clean_aug = equidistant_points(frequency = frequency_clean,
                                                              signal = field_clean)  # n_points defaults to 5e3
    # retrieving central carrier
    central_carrier = central_frequency(frequency = frequency_clean_aug, signal = field_clean_aug)
    frequency, field = torch.from_numpy(frequency_clean_aug), torch.from_numpy(field_clean_aug)

    laser = ComputationalLaser(
        # laser specific parameters
        frequency = frequency * 1e-12, 
        field = field, 
        central_frequency=central_carrier,
        # environment parametrization
        compressor_params = compressor_params,
        B=B_integral)
    
    return laser


class EnvParametrization: 
    """Parametrizes the default L1 Pump Environment."""
    def __init__(
            self, 
            compressor_params:torch.TensorType=default_compressor_params,
            lb:torch.TensorType=control_lower_bound,
            ub:torch.TensorType=control_upper_bound,
            B_integral:float=2
    ):
        """
        Env parameters.
        For an in-detail explanation of the whole model consider checking out this notebook:
        https://github.com/fracapuano/ELIopt/blob/main/notebooks/SemiPhysicalModel/SemiPhysicalModel_v2.ipynb. 

        Args:
            lb (torch.tensor, optional): Tensor representing the lower bound for the control parameters. 
                                         Defaults to control_lower_bound.
            ub (torch.tensor, optional): Tensor representing the upper bound for the control parameters. 
                                         Defaults to control_upper_bound.

        """
        # env parametrization
        # alpha_GDD, alpha_TOD, alpha_FOD
        self.compressor_params = compressor_params
        
        self.bounds = torch.vstack(
            (lb, ub)
        )
        # non-linear phase accumulation parameter - noisy estimate of real value
        self.B_integral = B_integral
    
    def get_parametrization(self)->List[torch.tensor]: 
        """Returns the env parametrization.
        
        Returns: 
            List[torch.tensor]: List of elements parametrizing the environment.
                                Ordered as [compressor_params, bounds, B_integral] 
        """
        return [
            self.compressor_params, 
            self.bounds, 
            self.B_integral
        ]

default_params = EnvParametrization()
_, default_bounds, *_ = default_params.get_parametrization()
default_om = torch.tensor((1e-24, 1e-36, 1e-48), dtype=torch.float64)

class ControlUtils: 
    def __init__(
            self, 
            om:torch.TensorType=default_om, 
            bounds:torch.TensorType=default_bounds): 
        """
        Args: 
            om (torch.tensor, optional): Order of magnitude of control parameters.
                                         Defaults to default_om (that is, controls in SI units).
            bounds (torch.tensor, optional): Bounds for control parameters. Default to default_params.
                                             (that is, default compressor of L1 Pump)
        
        """
        self.forward_om = om  # from non-SI units to SI units
        self.backward_om = 1 / om  # from SI units to non-SI units

        self.bounds = bounds * self.backward_om  # non SI units
    
    def scale_control(
            self,
            action:torch.TensorType,
        )->torch.TensorType: 
        """
        This function scales the input control `action` into output action.
        Normalization is performed with min-max normalization using bounds.
        
        Args: 
            action (torch.tensor): Action to be scaled using bounds. 
        
        Returns: 
            torch.tensor: Scaled action. Now, in the 0-1 range.
        """
        lower, upper = self.bounds
        return (action - lower) / (upper - lower) 

    def descale_control(
            self,
            action:torch.TensorType,
        )->torch.TensorType: 
        """
        This function descales the input control `action` into output action.
        Denormalization is performed inverting min-max normalization using bounds.
        
        Args: 
            action (torch.tensor): Action to be de-scaled using bounds. Each element is the 0-1 range 
                                   considered.
        
        Returns: 
            torch.tensor: Descaled action. Now, in the original range.
        """
        lower, upper = self.bounds
        return (upper - lower) * action + lower

    def controls_demagnify(self, input_control:torch.TensorType)->torch.TensorType:
        """
        This function applies a demagnification operation on input control. 
        Considering that each element is assumed to take values in a predefinite range, we can use
        this knowledge to retrieve the root of the number rather root and its actual order of magnitude. 
        
        This is done considering how largely small is the order or magnitude (om) of input control values.
        
        Args: 
            input_control (torch.tensor): Input control to the actual laser. Expressed in SI units.
        
        Returns: 
            torch.tensor: Output control, demagnified.
        """
        # SI -> non-SI
        return input_control * self.backward_om

    def control_magnify(self, input_control:torch.TensorType)->torch.TensorType:
        """
        This function applies a demagnification operation on input control. 
        Considering that each element is assumed to take values in a predefinite range, we can use
        this knowledge to retrieve the root of the number rather root and its actual order of magnitude. 
        
        This is done considering how largely small is the order or magnitude (om) of input control values.
        
        Args: 
            input_control (torch.tensor): Input control to the actual laser. Expressed in non-SI units.
        
        Returns: 
            torch.tensor: Output control, demagnified.
        """
        # non-SI -> SI
        return input_control * self.forward_om
    
    def demagnify_scale(self, input_control:torch.TensorType)->torch.TensorType:
        """
        Combines magnification and 0-1 scalig of input control. 
        Arguments and returns are the same as before.

        The control returned by this function would NOT be the one forwarded in the Computational Laser
        considered and will instead be used as observation by the agent.
        """
        # ex: 2.9e-22 -(demagnifies)-> 290 -(descales)-> 0.9  (fictional numbers)
        return self.scale_control(self.controls_demagnify(input_control))

    def remagnify_descale(self, input_control:torch.TensorType)->torch.TensorType:
        """
        Combines demagnification and descalig of input control. 
        Arguments and returns are the same as before.

        The control returned by this function would be the one forwarded in the Computational Laser
        considered.
        """
        # ex: 0.9 -(descaled)-> 290 -(magnified)-> 2.9e-22 -(des) (fictional numbers)
        return self.control_magnify(self.descale_control(input_control))
    