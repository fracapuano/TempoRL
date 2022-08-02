import numpy as np
def is_stalling(x:np.array, thresh:float=1e-6)->bool: 
    """
    This function takes an array of obj function (ideally coming from a small buffer of 5 to 10 observations) values in an optimization route
    as input and returns a boolean which is analysing whether or not the optimization route is stalling (in the sense that no significant im-
    provement in the objective function is done) with respect to a considered threshold.
    """
    # mean absolute improvement in considered buffer
    if np.mean(np.abs(np.diff(x))) <= thresh: 
        return True
    else: 
        return False