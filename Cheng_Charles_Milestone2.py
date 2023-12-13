# %%
"""
## ME 449 Mobile Manipulation Capstone: Milestone 2 ##
Author: Charles Cheng

If you are using an Anaconda environment, replace myenv with environment name and execute
> conda activate myenv

To run this file, navigate to the code directory and execute
> python Cheng_Charles_milestone2.py
"""

# %%
# Import libraries
import numpy as np # library for matrices
import pandas as pd # library for csv
import modern_robotics as mr 
import matplotlib.pyplot as plt # library for plotting

# %%
# Extracts px, py, pz vector from transformation matrix
def TransformPosition(T):
    return T[0:3, 3]

# Extracts rotation matrix from transformation matrix
def TransformRotation(T):
    return T[0:3, 0:3]

# Calculates minimum distance from one frame to another
def TrajectoryDistance(Tstart, Tend):
    return np.linalg.norm(TransformPosition(Tend) - TransformPosition(Tstart))

# Calculates minimum angle from one frame to another
def TrajectoryAngle(Tstart, Tend):
    Rse = TransformRotation(Tstart).T @ TransformRotation(Tend)
    return np.linalg.norm(mr.so3ToVec(mr.MatrixLog3(Rse)))

# Calculates minimum time needed to traverse from one frame to another
def SegmentDuration(Tstart, Tend, maxLinVel = 0.3, maxAngVel = 0.3):
    tMove = round(TrajectoryDistance(Tstart, Tend)/maxLinVel, 2)
    tRotate = round(TrajectoryAngle(Tstart, Tend)/maxAngVel, 2)
    return max(tMove, tRotate)

# %%
def TrajectoryGenerator(Tsei, Tsci, Tscf, Tceg, Tces, k):
    """
    TrajectoryGenerator returns a list of end-effector configurations needed to traverse 8-segment
    path from initial config to inital standoff config to initial grasp config to initial standoff
    config to final standoff config to final grasp config to final standoff config. Gripper takes
    time to open and close, so the end-effector remains stationary during those times. 
    
    :param Tsei:    initial configuration of the end-effector in the reference trajectory
    :param Tsci:    cube's initial configuration
    :param Tscf:    cube's desired final configuration 
    :param Tceg:    end-effector's configuration relative to the cube when it is grasping the cube
    :param Tces:    end-effector's standoff configuration above the cube, before and after grasping,
                    relative to the cube
    :param k:       number of trajectory reference configurations per 0.01 seconds
   
    :return trajTotal:   representation of the N configurations of the end-effector along the entire
                         concatenated eight-segment reference trajectory
    """
    
    # Configuration targets 
    T1 = Tsci @ Tces
    T2 = Tsci @ Tceg
    T5 = Tscf @ Tces
    T6 = Tscf @ Tceg
    eeTraj = [Tsei, T1, T2, T2, T1, T5, T6, T6, T5]
    
    # Time needed to traverse each segment
    gripTime = 0.9 # amount of time it takes gripper to open/close
    t1 = SegmentDuration(eeTraj[0], eeTraj[1])
    t2 = SegmentDuration(eeTraj[1], eeTraj[2])
    t3 = gripTime # gripper closing 
    t4 = SegmentDuration(eeTraj[3], eeTraj[4])
    t5 = SegmentDuration(eeTraj[4], eeTraj[5])
    t6 = SegmentDuration(eeTraj[5], eeTraj[6])
    t7 = gripTime # gripper opening
    t8 = SegmentDuration(eeTraj[7], eeTraj[8])
    timeVec = [t1, t2, t3, t4, t5, t6, t7, t8]
    
    isClosed = 0 # is gripper closed?
    trajTotal = [] # list of [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripperState]
    for i in range(8):
        segment = mr.CartesianTrajectory(eeTraj[i], eeTraj[i+1], timeVec[i], int(timeVec[i]*100*k), 5)
        if i == 2 or i == 6:
            isClosed ^= 1 # reverse gripper state in segments 3 and 7
        for config in segment:
            trajTotal.append([config[0][0], config[0][1], config[0][2], 
                              config[1][0], config[1][1], config[1][2], 
                              config[2][0], config[2][1], config[2][2], 
                              config[0][3], config[1][3], config[2][3],
                              isClosed])
    
    return trajTotal