import numpy as np

def rosen(x: np.array):
    """
    This function returns the (scalar) value of the generalized n-dimensional rosenbrock function at any point x of size (n,)
    """
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def gradient(f:function, x:np.array, mode:str="forward")->np.array: 
    """
    This function numerically computes the gradient of the given function. 
    
    """