import torch
from scipy.optimize import Bounds
import numpy as np
from typing import Tuple
from tqdm import tqdm

class LaserOptimization(torch.nn.Module): 
    """Custom Pytorch model for gradient based optimization.
    """
    def __init__(self, CL:object, loss:str="l1") -> None:
        """Init function.

        Args:
            CL (object): ComputationalLaser object implementing the 'forward_pass(x,)' method needed to predict the temporal profile
                         corresponding to the control x itself.
            loss (str): Type of loss to be used to compare the two profiles. Defaults to l1 norm with sum reduction. 
        """
        super().__init__()
        if loss.lower() == "l1": 
            self.loss = torch.nn.L1Loss(reduction = "sum")
        
        bounds_control = Bounds(
                    # GDD         # TOD          # FOD
            lb = (2.3522e-22, -1.003635e-34, 4.774465e-50),
            ub = (2.99624e-22, 9.55955e-35, 1.4323395e-49)
        )

        bounds_matrix = np.vstack((bounds_control.lb, bounds_control.ub)).T
        
        self.low_bound = torch.tensor(bounds_matrix[:, 0])
        self.up_bound = torch.tensor(bounds_matrix[:, 1])
        
        self.ComputationalLaser = CL
        
        # initialize weights with random numbers
        control = torch.distributions.Uniform(
            low = torch.from_numpy(bounds_matrix[:, 0]), 
            high = torch.from_numpy(bounds_matrix[:, 1])
        ).sample()

        # make weights torch parameters
        self.control = torch.nn.Parameter(control).cuda() if torch.cuda.is_available() else torch.nn.Parameter(control)    
        
    def objective_function(self, mu:float=1e-3) -> torch.tensor:
        """
        Implements the function to be minimised. In this case such a function will be the L1 norm corrected 
        with a log barrier to maintain the parameters into the feasible region.
        
        Args: 
            penalty (float): Penalty term. Defaults to 1e-3.
        
        Returns: 
            torch.tensor: One-element tensor (scalar) storing the value of objective function.
        """
        
        _, control_profile = self.ComputationalLaser.forward_pass(self.control)
        _, target_profile = self.ComputationalLaser.transform_limited()

        f_control = self.loss(control_profile, target_profile)
        constraint_violation = torch.log(self.control - self.low_bound).sum() + torch.log(self.up_bound - self.control).sum()

        return f_control + mu * constraint_violation

def optimization(model:object, optimizer:torch.optim, maxit:int=int(5e3)) -> Tuple[torch.tensor, torch.tensor]:
    """This function performs optimization (using optimizer) of the model defined in model (pytorch model)
    whose variables are nn.Parameters for a maximal number of iterations equal to maxit. 

    Args:
        model (object): Pytorch model characteristic of the machine considered.
        optimizer (torch.optim): Pytorch optimizers used (fully instantiated)
        maxit (int, optional): Number of iterations to carry out optimization. Defaults to int(5e3).

    Returns:
        Tuple[torch.tensor, torch.tensor]: Losses over iterate and best control parameters found, respectively. 
    """
    losses = torch.zeros(size = maxit)
    
    for it in (pbar := tqdm(range(maxit))):
        pbar.set_description("Iteration %s/%s" %it %maxit)
        # zero-grad optimizer
        optimizer.zero_grad()
        # backwarding objective function
        model.objective_function().backward()
        # stepping optimizer
        optimizer.step()
        # storing the value of the loss
        losses[it] = model.objective_function()
    
    return losses