from copy import copy
import numpy as np
import time
from pyrsistent import m
from scipy.optimize import rosen, rosen_der, rosen_hess
import matplotlib.pyplot as plt

def rosen(x: np.array):
    """
    This function returns the (scalar) value of the generalized n-dimensional rosenbrock function at any point x of size (n,)
    """
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def gradient(f, x:np.array, mode:str="f", h:float=1e-6)->np.array: 
    """This function computes the numerical gradient of "f" in "x" according to "mode".

    Args:
        f (function): The function whose gradient is to be obtained.
        x (np.array): The vector in which the function gradient should be obtained.
        mode (str, optional): The differentiation mode to use in the numerical differentiation setting; either "forward" (or "f") or "centered" (or "c"). 
                              "forward" is less precise than "centered", but it is less computationally expensive, whereas "centered" is more precise but
                              requires the double of function evaluations. Defaults to "forward".
        h (float, optional): The (small) scalar increment to be used in finite differentiation. The smaller h is the smaller the difference between numerical and
                             exact derivative gets. However, to a decrease in such "analytical error" corresponds an increase in "numerical error", so h should not
                             get too small. Defaults to 1e-6.

    Returns:
        np.array: The gradient of "f" in "x", grad_f(x).
    """
    # check on differentiation mode
    if mode.lower() not in ["forward", "centered", "f", "c"]: 
        raise ValueError("""Diff. mode not in ["forward", "centered", "f", "c"]. 
        Only these values are supported. Type "help(gradient)" in the command line for more information...
        """)
    
    fx = f(x)
    n = x.shape[0]
    idn = np.identity(n)

    grad_fx = np.zeros(shape = n)

    mode = mode.lower()
    # DEBUG purposes:
    # start_time = time.time()

    if mode == "f":  
        for i in range(n): 
            grad_fx[i] = (f(x + h*idn[:, i]) - fx)/h
    elif mode == "c": 
        for i in range(n): 
            grad_fx[i] = (f(x + h*idn[:, i]) - f(x - h*idn[:, i]))/(2*h)

    # DEBUG purposes: 
    # grad_time = time.time() - start_time
    # print(grad_time)

    return grad_fx

def hessian(f, x:np.array, h:float=np.sqrt(1e-8))->np.ndarray:
    """This function returns the hessian of "f" computed in "x"

    Args:
        f (function): The function whose hessian is to be obtained.
        x (np.array): The vector in which the function hessian should be obtained.
        h (float, optional):  The (small) scalar increment to be used in finite differentiation. The smaller h is the smaller the difference between numerical and
                             exact derivative gets. However, to a decrease in such "analytical error" corresponds an increase in "numerical error", so h should not
                             get too small. Defaults to np.sqrt(1e-8) (since there is a squaring operation of the increment in numerical hessian).
    Returns:
        np.ndarray: The hessian of "f" in "x", hess_f(x).
    """
    fx = f(x)
    n = x.shape[0]

    idn = np.identity(n)
    hess_fx = np.zeros_like(idn)
    
    # DEBUG purposes
    # start_time = time.time()

    for i in range(n): 
        # to reduce the number of function evaluation
        fx_incrementOnI = f(x + h*idn[:, i])
        for j in range(i, n): 
            # to reduce the number of function evaluation
        
            if j == i: # diagonal elements
                hess_fx[i, j] = (f(x + h*idn[:, i]) + f(x - h*idn[:, i]) - 2*fx)/h**2

            elif j != i: # off-diagonal elements
                hess_fx[i, j] = (f(x + h*idn[:, i] + h*idn[:, j]) - fx_incrementOnI - f(x + h*idn[:,j]) + f(x))/h**2
    
    # DEBUG purposes
    # hess_time = time.time() - start_time
    # print(hess_time)

    # filling the submatrix using the upper one
    hess_fx_copy = copy(hess_fx)
    # filling the diagonal of the submatrix matrix with zeros
    np.fill_diagonal(hess_fx_copy, 0)

    return hess_fx + hess_fx_copy.T

def test(n:int=10, n_test:int=10)->None: 
    """This function tests numeric approximation of hessian and gradient of the rosenbrock function using as reference solutions 
    the scipy implementation of gradient and hessian themselves. 

    Args:
        n (int, optional): Dimension of the space in which to conduct testing. Defaults to 10.
        n_test (int, optional): Number of tests to carry out with different vectors for hessian product. Defaults to 10.
    """
    
    hess_prod_err = np.zeros(n_test)
    grad_err = np.zeros(n_test)

    for i in range(n_test): 
        p_try = np.random.random(n)
        p_try_hess = np.random.random(n)

        grad_err[i] = np.linalg.norm(rosen_der(p_try) - gradient(rosen, p_try))
        hess_prod_err[i] = (np.linalg.norm((rosen_hess(p_try) - hessian(rosen, p_try)) @ p_try_hess))/np.linalg.norm(p_try_hess)

    print(f"Avg gradient error: {grad_err.mean()}")
    print(f"Avg hessian multiplication herror: {hess_prod_err.mean()}")

def computational_time()->None: 
    """This function tests performance by means of computational time of basic numerical differentiation using as reference solutions 
    the scipy implementation of gradient and hessian themselves. 

    Args:
        n_test (int, optional): Number of tests to carry out with different vectors for hessian product. Defaults to 10.
    """
    # different space dimensions to test
    dimensions = [3, 10, 25, 50, 100]
    
    num_gradient = np.zeros(len(dimensions))
    ex_gradient = np.zeros(len(dimensions))
    num_hess = np.zeros(len(dimensions))
    ex_hess = np.zeros(len(dimensions))

    i = 0
    for d in dimensions: 
        
        p_try = np.random.random(size = d)
        # numerical gradient
        s_time = time.time()
        gradient(rosen, p_try)
        computing_time = time.time() - s_time
        num_gradient[i] = computing_time
        
        # exact gradient
        s_time = time.time()
        rosen_der(p_try)
        computing_time = time.time() - s_time
        ex_gradient[i] = computing_time
        
        # numerical hessian
        s_time = time.time()
        hessian(rosen, p_try)
        computing_time = time.time() - s_time
        num_hess[i] = computing_time

        # exact hessian
        s_time = time.time()
        rosen_hess(p_try)
        computing_time = time.time() - s_time
        ex_hess[i] = computing_time

        i += 1
    
    # plotting the results obtained
    fig, ax = plt.subplots(ncols = 2)

    ax[0].plot(dimensions, num_gradient, label = "Numerical Gradient", ls = "--", lw = 1)
    ax[0].plot(dimensions, ex_gradient, label = "scipy Gradient", ls = "--", lw = 1)
    ax[1].plot(dimensions, num_hess, label = "Numerical Hessian", ls = "--", lw = 1)
    ax[1].plot(dimensions, ex_hess, label = "scipy Hessian", ls = "--", lw = 1)

    ax[0].scatter(dimensions, num_gradient)
    ax[0].scatter(dimensions, ex_gradient)
    ax[1].scatter(dimensions, num_hess)
    ax[1].scatter(dimensions, ex_hess)
    
    ax[0].legend()
    ax[0].set_xticks(dimensions)
    ax[0].set_yscale("log")
    ax[1].legend()
    ax[1].set_xticks(dimensions)
    ax[1].set_yscale("log")

    plt.show()
