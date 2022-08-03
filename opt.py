import numpy as np
from numpy.linalg import norm
from utils.function import *
from scipy.optimize import line_search
import matplotlib.pyplot as plt

def find_alpha(
    f, 
    grad_f, 
    xk:np.array,
    pk:np.array,
    alpha_0:float=0.5,  
    c1:float=1e-4, 
    rho:float=8e-1, 
    maxit_bt:int=10, 
    maxit_wolfe=50)->float: 
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
        maxit_wolfe (int, optional): Maximal number of iterations to verify strong Wolfe conditions. Defaults to 50.

    Returns:
        float: _description_
    """
    
    alpha_k = line_search(f, grad_f, xk, pk, maxiter = maxit_wolfe)[0]
    if alpha_k is None: # find alpha using regular Armijo backtrack
        alpha_k = alpha_0
        armijo_line = f(xk) - c1*alpha_k*(grad_f(xk) @ pk)
        bt_it = 0

        while f(xk - alpha_k*pk) > armijo_line and bt_it < maxit_bt: 
            alpha_k = rho*alpha_k
            bt_it = bt_it + 1
    
    return alpha_k

def pr_plus(
    f,
    grad_f, 
    x0:np.array, 
    tolgrad:float=1e-6, 
    maxit:int=int(1e4), 
    alpha_0:float=0.5,
    c1:float=1e-4, 
    rho:float=0.8, 
    maxit_bt:int=50,
    maxit_wolfe:int=50)->np.array:
    """This function implements the NonLinear Conjugate Gradient Method, in the Polak-Ribieré "+" variant proposed by Nocedal-Wright in
    "Numerical Optimization".

    Args:
        f (_type_): The objective function to optmize.
        grad_f (_type_): A function able to compute the gradient of the objective function to optimize.
        x0 (np.array): A starting point serving as an initial guess.
        tolgrad (float, optional): Tolerance on the norm of the gradient below which declaring convergence. Defaults to 1e-6.
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        alpha_0 (float, optional): Initial value for the steplength prior to backtracking
        c1 (float, optional): Slope of the Armijo line. Defaults to 1e-4.
        rho (float, optional): Backtracking factor used to scale down the steplength. Defaults to 0.8.
        maxit (int, optional): Maximal number of iterations in the iterative method considered. Defaults to int(1e4).
        maxit_wolfe (int, optional): Maximal number of iterations to satisfy strong Wolfe conditions. Defaults to 10.

    Returns:
        np.array: the optimal point to which the optimization route has converged. 
    """
    xk = x0
    grad_xk = grad_f(xk)

    SDOld = -grad_xk
    p0 = SDOld
    k = 0

    grads = np.zeros(maxit)
    grad_norm = norm(grad_xk)
    pk = p0
    
    while k < maxit and grad_norm >= tolgrad: 
        grad_xk = grad_f(xk)
        grads[k] = norm(grad_xk)

        SDNew = -grad_xk
        # "+" variation of regular Polak-Ribiere algorithm
        betaPR_k = max(0, (SDNew @ (SDNew - SDOld))/(SDOld @ SDOld))
        pk = SDNew + betaPR_k * pk
        # steplength selection
        alpha_k = find_alpha(f, grad_f, xk, pk, c1=c1, alpha_0=alpha_0, rho=rho, maxit_bt=maxit_bt, maxit_wolfe=maxit_wolfe)
        
        xk = xk + alpha_k * pk            

        SDOld = SDNew
        grad_norm = norm(SDNew)
        
        k += 1
    # "cutting" grads to the correct shape
    grads = grads[:k]
    return grads

def rosen_grad_handle(xk): 
    """Gradient function handle. Takes as input only the point to compute the gradient at.

    Args:
        xk (_type_): Point in which to compute the gradient.

    Returns:
        (function): Function handle considered.
    """
    return gradient(rosen, xk)

def plot_gradient_norm():
    """This function plots the gradient norm over iterations for diagnostic purposes. 
    """
    grads = pr_plus(rosen, rosen_grad_handle, np.random.random(3), maxit_bt=10, rho=3e-1, alpha_0 = 1)
    _, ax = plt.subplots()

    ax.plot(grads)
    ax.set_xlabel("Number of iterations", fontdict = {"fontsize": 12})
    ax.set_ylabel("log(Norm of the gradient)", fontdict = {"fontsize": 12})

    ax.set_yscale("log")
    ax.set_title("Norm of the gradient over iterationts", fontdict = {"fontweight":"bold", "fontsize": 14})
    
    ax.grid()

    plt.show()

def main(): 
    plot_gradient_norm()

if __name__ == "__main__": 
    main()