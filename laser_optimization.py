from utils import physics, optimization 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import time 

import warnings
warnings.filterwarnings("ignore")

def main()->None: 
    file_name = "LLNL_160809_freq.csv"
    df = pd.read_csv("data/"+file_name, header = None)
    df.columns = ["Frequency", "Wavelength", "Intensity", "Phase", "Phase (cutted)"] 

    # selecting only the interesting elements for the analysis
    frequency = df.loc[:, "Frequency"].values * 10**12 #THz to Hz
    intensity = df.loc[:, "Intensity"].values

    laser = physics.Laser(frequency=frequency, intensity=intensity)

    loss = lambda control: mse(laser.target_pulse(), laser.control_to_pulse(control))

    grad_handle = lambda control: optimization.gradient(loss, control)
    hess_handle = lambda control: optimization.hessian(loss, control)

    start_time = time.time()
    grad_handle(10*np.random.random(size = 3))
    grad_time = time.time() - start_time

    start_time = time.time()
    hess_handle(10*np.random.random(size = 3))
    hess_time = time.time() - start_time

    print("time for one gradient: {:.3e}".format(grad_time))
    print("time for one hessian: {:.3e}".format(hess_time))

    #control_start = 0.5 * np.array([27.0, 40.0, 50.0])

    #solution, grads = optimization.newton(loss, grad_handle, hess_handle, control_start, maxit = 85)
    
    #print(solution)
    #plt.plot(grads)
    #plt.show()

    optimal_control = np.array([68.52904811, 732.09111505, 3378.57667325])
    plt.plot(laser.target_pulse(), label = "target") 
    plt.plot(laser.control_to_pulse(optimal_control), label = "obtained")
    plt.legend()
    plt.show()

if __name__=="__main__": 
    main()
    