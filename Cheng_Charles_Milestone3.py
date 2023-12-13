# %%
"""
## ME 449 Milestone 3 ##
Author: Charles Cheng
"""

# %%
# Import libraries
import numpy as np # library for matrices
import pandas as pd # library for csv
import modern_robotics as mr 
import matplotlib.pyplot as plt # library for plotting

# %%
def FeedBackControl(thetalist, X, Xd, Xdnext, Kp, Ki, intXerr, dt):
    """
    FeedBackControl returns commanded wheel and arm joint speeds
    and numerical integral error in a list. 
    
    :param thetalist: list of joint configurations
    :param X:         current actual ee configuration (Tse)
    :param Xd:        current ee reference configuration (Tsed)
    :param Xdnext:    ee reference configuration at time dt later (Tsednext)
    :param Kp:        P gain matrix
    :param Ki:        I gain matrix
    :param intXerr:   numerical integral error
    :param dt:        timestep (units: s)
    
    :return udtheta:  commanded wheel and arm joint speeds
    :return Xerr:     error
    :return intXerr:  numerical integral error
    """
    Xerr = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
    intXerr = intXerr + dt*Xerr
    Vd = 1/dt*mr.se3ToVec(mr.MatrixLog6(mr.TransInv(Xd) @ Xdnext))
    V = mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd + Kp @ Xerr + Ki @ intXerr
    
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
    lplusw = 0.385 # length + width of mobile base
    r = 0.0475 # radius of wheels
    F = r/4 * np.array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [-1/lplusw, 1/lplusw, 1/lplusw, -1/lplusw], 
                        [1, 1, 1, 1], 
                        [-1, 1, -1, 1],
                        [0, 0, 0, 0]])
    T0e = mr.FKinBody(M0e, Blist, thetalist[3:])
    
    Ja = mr.JacobianBody(Blist, thetalist[3:]) # Jacobian of arm
    Jb = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0)) @ F # Jacobian of base
    Je = np.hstack((Jb, Ja)) # combined Jacobian 
    
    udtheta = np.linalg.pinv(Je, rcond=1e-3) @ V
    return [udtheta, Xerr, intXerr]