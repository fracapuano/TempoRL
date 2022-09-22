import numpy as np
from numpy.linalg import norm
from utils.numerical_diff import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from typing import Tuple
import torch
from torch.linalg import norm
from tqdm import tqdm

def find_alpha(
    f, 
    grad_f, 
    xk:np.array,
    pk:np.array,
    alpha_0:float=1,  
    c1:float=1e-4, 
    rho:float=8e-1, 
    maxit_bt:int=10)->float: 
    """This function finds the optimal steplenght using a combination of Wolfe conditions and (in case these conditions could not be matched) and
    regular Armijo conditions using backtrack.

    Args:
        f (function): Function handle of the objective function to optimize
        grad_f (function): Function handle of the gradient of the objective function to optimize
        xk (np.array): The point at the k-th iterate that is being considered
        pk (np.array): The search direction found
        alpha_0 (float, optional): The initial value of the steplength. Used in backtracking with Armijo as starting point. Defaults to 0.5.
        c1 (float, optional): Slope of the Armijo line. Defaults to 1e-4.
        rho (float, optional): Backtracking factor used to scale down the steplength. Defaults to 8e-1.
        maxit_bt (int, optional): Maximal number of backtrack iterations. Defaults to 10.

    Returns:
        float: optimal steplength according to line search
    """    
    # find alpha using regular Armijo backtrack
    armijo_line = lambda alpha_k: f(xk - alpha_k * pk) - f(xk) - c1*alpha_k*(grad_f(xk) @ pk)
    alpha_k = alpha_0
    bt_it = 0

    while armijo_line(alpha_k) > 0 and bt_it < maxit_bt: 
        alpha_k = rho*alpha_k
        bt_it = bt_it + 1
    
    return alpha_k

def newton(
    f, 
    grad_f, 
    hess_f, 
    x0:np.array,
    tolgrad:float=1e-6, 
    maxit:int=int(1e4), 
    alpha_0:float=1, 
    c1:float=1e-4, 
    rho:float=8e-1, 
    maxit_bt:int=10,
    gmres_maxit:int=int(1e2), 
    verbose:int=0)->Tuple[np.array, np.array]:
    """This function implements the Newton Method for non linear optimization using a mixture of strong Wolfe conditions 
    and Armijo backtracking to carry out linesearch. 

    Args:
        f (function): Objective function to optimize.
        grad_f (function):  A function able to compute the gradient of the objective function to optimize.
        hess_f (function):  A function able to compute the hessian of the objective function to optimize.
        x0 (np.array): A starting point serving as an initial guess.
        tolgrad (float, optional): Tolerance on the norm of the gradient below which declaring convergence. Defaults to 1e-6.
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        alpha_0 (float, optional):  Initial value for the steplength prior to backtracking. Defaults to 1.
        c1 (float, optional): Slope of the Armijo line. Defaults to 1e-4.
        rho (float, optional): Backtracking factor used to scale down the steplength. Defaults to 8e-1.
        maxit_bt (int, optional): Maximal number of backtrack iterations to satisfy Armijo condition. Defaults to int(50).
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        verbose (int, optional): Verbose parameter. Printing optimization route information if verbose is greater than 0. 
    Returns:
        Tuple[np.array, np.array]: 
            x_star: the optimal point to which the optimization route has converged
            grads: the norm of the gradient over the optimization route as well
    """
    xk = x0
    fxk = f(xk)
    grad_xk = grad_f(xk)
    grad_norm = norm(grad_xk)
    grads = np.zeros(maxit)

    k = 0
    while k < maxit and grad_norm > tolgrad: 
        it_time = time.time()
        xold = xk
        alpha_k = alpha_0
        hess_xk = hess_f(xk)
        grad_xk = grad_f(xk)
        fxk = f(xk)
        grad_norm = norm(grad_xk)
        grads[k] = grad_norm

        pk, _ = gmres(hess_xk, -grad_xk, maxiter=gmres_maxit)     
        alpha_k = find_alpha(f, grad_f, xk, pk, c1=c1, alpha_0=alpha_0, rho=rho, maxit_bt=maxit_bt)

        xk = xk + alpha_k * pk
        k += 1
        if verbose > 0:
            print("It {} - gradnorm {:.4e} - function value {:.4e} - solution difference {:.4e} - obtained in {} s".format(k, grad_norm, fxk, norm(xk - xold), time.time()-it_time))

    # "cutting" grads to the correct shape
    grads = grads[:k]
    print(f"Number of its: {k}")
    return xk, grads

def pr_plus(
    f,
    grad_f, 
    x0:np.array, 
    tolgrad:float=1e-6, 
    maxit:int=int(1e4), 
    alpha_0:float=1,
    c1:float=1e-4, 
    rho:float=0.8, 
    maxit_bt:int=10, 
    verbose:int=0)->Tuple[np.array, np.array]:
    """This function implements the NonLinear Conjugate Gradient Method, in the Polak-Ribieré "+" variant proposed by Nocedal-Wright in
    "Numerical Optimization".

    Args:
        f (function): The objective function to optmize.
        grad_f (function): A function able to compute the gradient of the objective function to optimize.
        x0 (np.array): A starting point serving as an initial guess.
        tolgrad (float, optional): Tolerance on the norm of the gradient below which declaring convergence. Defaults to 1e-6.
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        alpha_0 (float, optional): Initial value for the steplength prior to backtracking. Defaults to 0.5.
        c1 (float, optional): Slope of the Armijo line. Defaults to 1e-4.
        rho (float, optional): Backtracking factor used to scale down the steplength. Defaults to 0.8.
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        verbose (int, optional): Verbose parameter. Printing optimization route information if verbose is greater than 0. 

    Returns:
        Tuple[np.array, np.array]: 
            x_star: the optimal point to which the optimization route has converged
            grads: the norm of the gradient over the optimization route as well
    """
    xk = x0
    grad_xk = grad_f(xk)

    SDOld = -grad_xk
    p0 = SDOld
    k = 0

    grads = np.zeros(maxit)
    grad_norm = norm(grad_xk)
    pk = p0
    
    while k < maxit and grad_norm > tolgrad:
        it_time = time.time()
        xold = xk 
        grad_xk = grad_f(xk)
        grads[k] = norm(grad_xk)

        SDNew = -grad_xk
        # "+" variation of regular Polak-Ribiere algorithm
        betaPR_k = max(0, (SDNew @ (SDNew - SDOld))/(SDOld @ SDOld))
        pk = SDNew + betaPR_k * pk
        # steplength selection
        alpha_k = find_alpha(f, grad_f, xk, pk, c1=c1, alpha_0=alpha_0, rho=rho, maxit_bt=maxit_bt)
        
        xk = xk + alpha_k * pk           

        SDOld = SDNew
        grad_norm = norm(SDNew)
        
        k += 1
        if verbose > 0: 
            print("It {} - gradnorm {:.4e} - function value {:.4e} - solution difference {:.4e} - obtained in {} s".format(k, grad_norm, f(xk), norm(xk - xold), time.time()-it_time))
    # "cutting" grads to the correct shape
    grads = grads[:k]
    return xk, grads

def rosen_grad_handle(xk:np.array): 
    """Gradient function handle. Takes as input only the point to compute the gradient at.

    Args:
        xk (np.array): Point in which to compute the gradient.

    Returns:
        (function): Function handle considered.
    """
    return gradient(rosen, xk)

def rosen_hess_handle(xk:np.array): 
    """Hessian function handle. Takes as input only the point to compute the hessian at.

    Args:
        xk (np.array): Point in which to compute the hessian.

    Returns:
        (function): Function handle considered.
    """
    return hessian(rosen, xk)

def plot_gradient_norm(dim:int=10):
    """This function plots the gradient norm over iterations for diagnostic purposes.
    
    Args:
        dim (int): Dimension of the space considered.
    """
    x_start = np.random.random(dim)
    tolgrad = 1e-6

    print("Starting newton method...")
    start_time = time.time()
    x_newton, grads_n = newton(rosen, rosen_grad_handle, rosen_hess_handle, x_start, tolgrad=tolgrad, verbose = 1)
    completion = time.time() - start_time
    print("Newton complete! (elapsed in {:.3f} s)".format(completion))

    print("Starting PR+ method...")
    start_time = time.time()
    x_pr, grads_pr = pr_plus(rosen, rosen_grad_handle, x_start, maxit_bt=10, rho=0.5, alpha_0 = 1, tolgrad=tolgrad, verbose = 1)
    completion = time.time() - start_time
    print("PR+ complete! (elapsed in {:.3f} s)".format(completion))

    print("Differnce in the found solutions is equal to {:.6e}".format(norm(x_pr - x_newton)))
    
    _, ax = plt.subplots()

    ax.plot(grads_pr, label = "PR+")
    ax.plot(grads_n, label = "Newton")
    ax.set_xlabel("Number of iterations", fontdict = {"fontsize": 12})
    ax.set_ylabel("log(Norm of the gradient)", fontdict = {"fontsize": 12})

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_title("Norm of the gradient over iterations", fontdict = {"fontweight":"bold", "fontsize": 14})
    
    ax.grid()
    ax.legend()

    plt.show()

class PenalizedFunction: 
    def __init__(self, f, bounds:torch.tensor, penalty_term:torch.tensor):
        """Init function takes only penalty term (to be updated during optimisation) and bounds.
        """
        self.f = f
        self.bounds = bounds
        self.epsilon = penalty_term
    
    def evaluate(self, x:torch.tensor) -> torch.tensor: 
        """
        This function returns the penalized version of the objective function considering input bounds. 
        """
        # minus(lb)-plud(ub) the input array
        pm_input = torch.hstack(list(map(lambda component: torch.stack((-component, component)), x)))
        # reshaping so as to have lower bound/upper bound/lower bound/...
        bounds = self.bounds.reshape(-1)
        # multiplying so as to have +lower bound/-upper bound/+lower bound/ ...
        multipliers = torch.empty_like(bounds)
        multipliers[::2] = 1; multipliers[1::2] = -1
        bounds = bounds * multipliers
        # summing component-wise input and corresponding bound to obtain square violation
        violations = torch.max(torch.zeros_like(bounds), pm_input + bounds) ** 2
        return self.f(x) + (1/self.epsilon) * violations.sum()
    
    def evaluate_root(self, x:torch.tensor) -> torch.tensor: 
        """
        This function returns the penalized version of the objective function considering input bounds. 
        """
        # minus(lb)-plud(ub) the input array
        pm_input = torch.hstack(list(map(lambda component: torch.stack((-component, component)), x)))
        # reshaping so as to have lower bound/upper bound/lower bound/...
        bounds = self.bounds.reshape(-1)
        # multiplying so as to have +lower bound/-upper bound/+lower bound/ ...
        multipliers = torch.empty_like(bounds)
        multipliers[::2] = 1; multipliers[1::2] = -1
        bounds = bounds * multipliers
        # summing component-wise input and corresponding bound to obtain violation
        violations = torch.max(torch.zeros_like(bounds), pm_input + bounds)
        return self.f(x) + (1/self.epsilon) * violations.sum()
      
    def optimize(self, 
                 x0:torch.tensor, 
                 maxit:int=int(1e3), 
                 tolgrad:float=1e-6, 
                 eta_0:float=1,
                 eps:float=1e-3,
                 square:bool=True) -> Tuple[torch.tensor, torch.tensor]: 
        """
        This function optimises the unconstrained (penalized) objective function
        """
        # the objective function is the penalized one 
        f = self.evaluate if square else self.evaluate_root
        k = 0
        xk = x0.requires_grad_(True)
        # giving gradient to xk
        f(xk).backward()

        points = torch.zeros(size = (maxit, len(xk)))
        grads = torch.zeros(maxit)

        grad_k = xk.grad
        
        sum_grad_squared = grad_k ** 2
        lr = eta_0 / torch.sqrt(eps + sum_grad_squared)

        for k in (pbar := tqdm(range(maxit))): 
            # store & display information
            points[k, :] = xk.detach()
            grads[k] = norm(grad_k)
            pbar.set_description("It. {} - f(xk) value {:4e}".format(k+1, f(xk).detach().numpy()))
            # update position
            xk = (xk.double() - lr.double() * grad_k.double()).detach().requires_grad_(True)
            # compute gradient at new position
            f(xk).backward()
            grad_k = xk.grad
            # adapt learning rate
            sum_grad_squared += grad_k ** 2
            lr = eta_0 / torch.sqrt(eps + sum_grad_squared)
            # reiterate (if not converged)
            if norm(grad_k) <= tolgrad: 
                points = points[:, :k]
                grads = grads[:k]
                break
            else: 
                k += 1
                # increasing the penalty as iterate progress
                self.epsilon *= .9999
                continue
        return points.detach(), grads

def main(): 
    plot_gradient_norm(dim=2)

if __name__ == "__main__": 
    main()