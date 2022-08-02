import numpy as np
import time

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
    
    mode = mode.lower()
    if mode == "f": 
        pass
    elif mode == "c": 
        pass

def hessian(f, x:np.array, h:float=np.sqrt(1e-6))->np.ndarray:
    """This function returns the hessian of "f" computed in "x"

    Args:
        f (function): The function whose hessian is to be obtained.
        x (np.array): The vector in which the function hessian should be obtained.
        h (float, optional):  The (small) scalar increment to be used in finite differentiation. The smaller h is the smaller the difference between numerical and
                             exact derivative gets. However, to a decrease in such "analytical error" corresponds an increase in "numerical error", so h should not
                             get too small. Defaults to np.sqrt(1e-6) (since there is a squaring operation of the increment in numerical hessian).
    Returns:
        np.ndarray: The hessian of "f" in "x", hess_f(x).
    """
    fx = f(x)

    n = x.shape[0]
    idn = np.identity(n)
    hess_fx = np.zeros_like(idn)
    
    #DEBUG purposes: start_time = time.time()

    for i in range(n): 
        # to reduce the number of function evaluation
        fx_incrementOnI = f(x + h*idn[:, i])
        for j in range(i, n): 
            # to reduce the number of function evaluation
            fx_incrementOnJ = f(x + h*idn[:,j])

            if j == i: # diagonal elements
                hess_fx[i, j] = (f(x + h*idn[:, i]) + f(x - h*idn[:, i]) - 2*fx)/h**2

            elif j != i: # off-diagonal elements
                hess_fx[i, j] = (f(x + h*idn[:, i] + h*idn[:, j]) - fx_incrementOnI - fx_incrementOnJ + f(x))/h**2
    
    #DEBUG purposes: hess_comp_time = time.time() - start_time
    
    return hess_fx

x_try = np.random.random(10)
print(hessian(rosen, x_try))