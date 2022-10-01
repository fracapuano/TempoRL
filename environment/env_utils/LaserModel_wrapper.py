import sys
import torch
import numpy as np
sys.path.append("../..")

from utils.physics import *
from utils.LaserModel_torch import ComputationalLaser as CL
from utils.LaserModel_torch import instantiate_laser

def descale_control(control:torch.tensor, given_bounds:torch.tensor, actual_bounds:torch.tensor)->torch.tensor: 
    r"""This function minmax-descales the control array in the range of "given bounds" $[a, b]$ to the "actual bounds" $[min, max]$ range.
    Descaling follows the equation $\min + \frac{(control - a)(\max - \min)}{(b-a)}$.

    Args:
        control (torch.tensor): Control to be descaled. (m,) shape.
        given_bounds (torch.tensor): Scale of the control given. mx2 shape, where m is the dimension of the given control.
        actual_bounds (torch.tensor): Scale of the desidered control. mx2 shape, where m is the dimension of the given control.

    Returns:
        torch.tensor: Descaled control.
    """
    a, b, min_, max_ = given_bounds[:,0], given_bounds[:,1], actual_bounds[:,0], actual_bounds[:,1]
    return min_ + ((control - a)*(max_ - min_)/(b-a))

def rescale_embedding(state:np.array)->np.array: 
    """This function converts a state embedding into a practical observation.

    Args:
        state (np.array): State embedded.

    Returns:
        np.array: Observation.
    """                     # PI     FWHM                FW25M                FW75M                AREA                BUILDUP
    mul_costants = np.array([1e-14, 1e+12, 1e+13, 1e+13, 1e+12, 1e+12, 1e+12, 1e+13, 1e+13, 1e+13, 1e+12, 1e+13, 1e+12, 1e+12, 1e+12, 1e+12])
    # multiplication aimed at having computationally stable numbers
    return mul_costants * state

def descale_embedding(embedding:np.array)->np.array: 
    """This function converts a state embedding into a practical observation.

    Args:
        embedding (np.array): State embedded.

    Returns:
        np.array: Observation in the actual unit of measures considered.
    """                         # PI     FWHM                FW25M                FW75M                AREA                BUILDUP
    mul_costants = 1 / np.array([1e-14, 1e+12, 1e+13, 1e+13, 1e+12, 1e+12, 1e+12, 1e+13, 1e+13, 1e+13, 1e+12, 1e+13, 1e+12, 1e+12, 1e+12, 1e+12])
    # multiplication aimed at having computationally stable numbers
    return mul_costants * embedding


class LaserWrapper: 
    def __init__(self, a = -5, b = 5, w0:float=12e-3, E:float=220e-3, min_thresh:float=1e-3):
        self.LaserModel = instantiate_laser()
        self.PulseEmbedder = PulseEmbedding(w0=w0, E=E, min_thresh=min_thresh)
        self.a = a; self.b = b
        
        self.bounds_SI = torch.from_numpy(np.array([
            [2.3522e-22, 2.99624e-22],
            [-1.003635e-34, 9.55955e-35],
            [4.774465e-50, 1.4323395e-49]])
        )

        self.bounds_control = torch.from_numpy(np.array([
            [self.a,self.b],
            [self.a,self.b],
            [self.a,self.b]
        ])
        )

    def transform_limited_embedding(self)->pd.Series: 
        """This function returns the Series corresponding to transform limited pulse.
        """
        time, pulse = self.LaserModel.transform_limited()
        # when cuda is available .numpy() only works on CPU
        if torch.cuda.is_available(): 
            time = time.cpu()
            pulse = pulse.cpu()
        
        values = rescale_embedding(self.PulseEmbedder.embed(time = time.numpy(), pulse = pulse.numpy()))
        index = self.PulseEmbedder.basic_index()
        return pd.Series(data = values, index = index)
        
    def forward_pass(self, control:torch.tensor)->pd.Series: 
        """This function performs a embedding on the forward pass of the laser model. The control is given as numbers in a custom range
        and translated to 

        Args:
            control (torch.tensor): Control parameters in the bounds_control range

        Returns:
            pd.Series: Forward Pass of the model
        """
        # sanity check
        if torch.prod(control) != torch.prod(torch.clamp(input = control, min = self.bounds_control[:,0], max = self.bounds_control[:,1])): 
            raise ValueError("Input control out of scaling bounds! Check controls consistency")
        # descaling provided control
        control_descaled = descale_control(control=control, given_bounds=self.bounds_control, actual_bounds=self.bounds_SI)
        # forward pass - with descaled control
        time, pulse = self.LaserModel.forward_pass(control_descaled)
        # when cuda is available .numpy() only works on CPU
        if torch.cuda.is_available(): 
            time = time.cpu()
            pulse = pulse.cpu()
        time, pulse = time.numpy(), pulse.numpy()
        # embedding forward pass 
        output = self.PulseEmbedder.embed(time = time, pulse = pulse)
        
        return rescale_embedding(output)