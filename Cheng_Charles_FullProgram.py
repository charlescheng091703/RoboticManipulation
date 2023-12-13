# %%
"""
## ME 449 Full Program ##
Author: Charles Cheng

Instructions of Use

If you are using an Anaconda environment, replace myenv with environment name and execute
>> conda activate myenv

To run this file, navigate to the code directory and execute one of the following in the command line. 
>> python Cheng_Charles_FullProgram.py best
>> python Cheng_Charles_FullProgram.py overshoot
>> python Cheng_Charles_FullProgram.py newTask
"""

# %%
# Import libraries
import numpy as np # library for matrices
import pandas as pd # library for csv
import modern_robotics as mr 
import matplotlib.pyplot as plt # library for plotting
import sys # library for parsing command line arguments 

# %%
# Import functions from other scripts
from Cheng_Charles_Milestone1 import NextState, MobileBaseConfig
from Cheng_Charles_Milestone2 import TrajectoryGenerator
from Cheng_Charles_Milestone3 import FeedBackControl

# %%
# Vector of config -> transformation matrix 
def ReconstructTransMat(configArr):
    return np.array([[configArr[0], configArr[1], configArr[2], configArr[9]],
                     [configArr[3], configArr[4], configArr[5], configArr[10]],
                     [configArr[6], configArr[7], configArr[8], configArr[11]],
                     [0, 0, 0, 1]])
    
def CubeConfig(qvec):
    return np.array([[np.cos(qvec[2]), -np.sin(qvec[2]), 0, qvec[0]], 
                     [np.sin(qvec[2]), np.cos(qvec[2]), 0, qvec[1]], 
                     [0, 0, 1, 0.025], 
                     [0, 0, 0, 1]])

# %%
# Given robot geometry
Tb0 = np.array([[ 1, 0, 0, 0.1662], # offset of chasis and arm base
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0.0026],
                [ 0, 0, 0, 1]])
M0e = np.array([[ 1, 0, 0, 0.033], # home config
                [ 0, 1, 0, 0],
                [ 0, 0, 1, 0.6546],
                [ 0, 0, 0, 1]])
Blist = np.array([[0, 0, 1, 0, 0.033, 0], # screw axes of arm joints
                 [0, -1, 0, -0.5076, 0, 0], 
                 [0, -1, 0, -0.3526, 0, 0], 
                 [0, -1, 0, -0.2176, 0, 0], 
                 [0, 0, 1, 0, 0, 0]]).T

# Initial, standoff, and grasp end-effector config
Tsei = np.array([[0, 0, 1, 0], 
                 [0, 1, 0, 0],
                 [-1, 0, 0, 0.5],
                 [0, 0, 0, 1]])
eeTilt = (2**0.5)/2
Tceg = np.array([[-eeTilt, 0, eeTilt, 0], 
                 [0, 1, 0, 0],
                 [-eeTilt, 0, -eeTilt, 0],
                 [0, 0, 0, 1]])
Tces = np.array([[-eeTilt, 0, eeTilt, 0], 
                 [0, 1, 0, 0],
                 [-eeTilt, 0, -eeTilt, 0.1],
                 [0, 0, 0, 1]])

# Initialization
currentConfig = np.array([-np.pi/6, -0.15, 0.14, 0, 0.3, -0.7, -1.6, 0, 0, 0, 0, 0, 0])
intXerr = np.zeros(6) # integral of error
configMat = []
XerrMat = []
k = 10 # number of configs per 0.01 seconds
dt = 0.01 # stepsize

# %%
if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        if filename == "best":
            print("Selected best parameters.")
            Kp = 0.6*np.eye(6) # P gain
            Ki = 0.001*np.eye(6) # I gain
            Tsci = CubeConfig([1, 0, 0]) # initial cube config
            Tscf = CubeConfig([0, -1, -np.pi/2]) # final cube config
        elif filename == "overshoot":
            print("Selected overshoot parameters.")
            Kp = np.eye(6) 
            Ki = 1.5*np.eye(6) 
            Tsci = CubeConfig([1, 0, 0]) 
            Tscf = CubeConfig([0, -1, -np.pi/2])
        elif filename == "newTask":
            print("Selected newTask parameters.")
            Kp = 0.4*np.eye(6) 
            Ki = 0.001*np.eye(6) 
            Tsci = CubeConfig([0.8, 0.2, -np.pi/6])
            Tscf = CubeConfig([-0.2, -0.8, -np.pi/4]) 
        else:
            raise Exception("Invalid selection. Choose best, overshoot, or newTask.")
    else:
        raise Exception("Incorrect number of arguments. Please only enter 1 (best, overshoot, or newTask).")
    
    # Generate reference trajectory
    refTraj = TrajectoryGenerator(Tsei, Tsci, Tscf, Tceg, Tces, k)
    
    # Apply feedforward and feedback control based on actual config
    for i in range(len(refTraj)-1):
        actualTse = MobileBaseConfig(currentConfig[:3]) @ Tb0 @ mr.FKinBody(M0e, Blist, currentConfig[3:8])
        Tsed = ReconstructTransMat(refTraj[i][:-1])
        Tsednext = ReconstructTransMat(refTraj[i+1][:-1])
        [controls, Xerr, intXerr] = FeedBackControl(currentConfig[:8], actualTse, Tsed, Tsednext, Kp, Ki, intXerr, dt)
        currentConfig = np.hstack((NextState(currentConfig[:-1], controls, dt, 2), refTraj[i+1][-1]))
        if i != 0 and i % k == 0:
            configMat.append(currentConfig)
            XerrMat.append(Xerr)
            
    # Converts numpy array of joint configurations and errors into csv files
    print("Generating animation csv file.")
    pd.DataFrame(configMat).to_csv(filename + '_config.csv', index=False, header=False)
    print("Writing error plot data.")
    pd.DataFrame(XerrMat).to_csv(filename + '_error.csv', index=False, header=False)
    
    # Plots error and saves it as an image
    print("Generating plot of error data.")
    XerrMat = np.array(XerrMat)
    for i in range(6):
        plt.plot(XerrMat[:, i])
    plt.title('X Error vs Time (ms)')
    plt.xlabel('Time (ms)')
    plt.ylabel('X Error')
    plt.legend(['thetabx', 'thetaby', 'thetabz', 'xbx', 'xby', 'xbz'])
    plt.savefig(filename + '_errorplot.png', format='png')
    print("Done.")

# %%
