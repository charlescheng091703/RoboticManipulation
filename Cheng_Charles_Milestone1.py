# %%
"""
## ME 449 Milestone 1 ##
Author: Charles Cheng
"""

# %%
# Import libraries
import numpy as np # library for matrices
import pandas as pd # library for csv
import modern_robotics as mr 
import matplotlib.pyplot as plt # library for plotting

# %%
def MobileBaseConfig(qvec):
    return np.array([[np.cos(qvec[0]), -np.sin(qvec[0]), 0, qvec[1]], 
                     [np.sin(qvec[0]), np.cos(qvec[0]), 0, qvec[2]], 
                     [0, 0, 1, 0.0963], 
                     [0, 0, 0, 1]])

# %%
def NextState(thetalist, dthetalist, dt, wmax):
    """
    NextState returns a vector of configurations at the next timestep.
    
    :param thetalist:   a 12-vector of robot's current config (units: rad)
    :param dthetalist:  a 9-vector of wheel and joint speeds (units: rad/s)
    :param dt:          timestep (units: s)
    :param wmax:        maximum angular speed of joints and wheels (units: rad/s)
    
    :return thetaNext:  a 12-vector of robot's next config dt later (units: rad)
    """
    # Apply speed limits
    for i, dtheta in enumerate(dthetalist):
        if dtheta > wmax:
            dthetalist[i] = wmax
        elif dtheta < -wmax:
            dthetalist[i] = -wmax
    
    nextJoint = thetalist[3:8] + dt*dthetalist[4:]
    nextWheel = thetalist[8:] + dt*dthetalist[:4]
    
    # Calculate new chasis config using odometry
    wheelDisp = nextWheel - thetalist[8:]
    lplusw = 0.385 # length + width of mobile base
    r = 0.0475 # radius of wheels
    F = r/4 * np.array([[-1/lplusw, 1/lplusw, 1/lplusw, -1/lplusw], 
                        [1, 1, 1, 1], 
                        [-1, 1, -1, 1]])
    Vb6 = np.hstack(([0, 0], F @ wheelDisp, 0))
    Tb1 = mr.MatrixExp6(mr.VecTose3(Vb6))
    Ts1 = MobileBaseConfig(thetalist[:3]) @ Tb1
    nextChasis = np.array([np.arccos(Ts1[0, 0]), Ts1[0, 3], Ts1[1, 3]])
    
    return np.hstack((nextChasis, nextJoint, nextWheel))
