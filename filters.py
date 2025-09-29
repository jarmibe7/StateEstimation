"""
run.py

Filters for HW0 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 09/19/2025
""" 
import numpy as np
import pandas as pd
import os

def normalize_angle(theta):
    return ((theta + np.pi) % (2*np.pi)) - np.pi

def dead_reckon(u_traj, motion_model, x0, t0, tf, h, Q=None, tspan=None, tsync='const'):
    """
    Function that executes a control trajectory given motion and measurement models, without filtering.

    Args:
        u_traj: Control trajectory
        motion_model: Motion model function of a robot
        x0: Initial state
        t0: Simulation start time
        tf: Simulation end time
        h: Timestep
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
    """
    # Initialization
    if tspan is None: tspan = np.arange(start=t0, stop=tf, step=h)
    assert tspan.shape[0] == u_traj.shape[0]
    x = x0
    sim = np.zeros((len(tspan), len(x0)))
    prev_time = tspan[0] - h
    prev_control = u_traj[0]
    for (i, t), u in zip(enumerate(tspan), u_traj):
        if tsync == 'const':
            # Control signals are not logged at a fixed timestep.
            # We can simulate at a fixed timestep, but send commands at proper times. In this implementation
            # the previous command is held onto, and used if it is still commanded at the current t.
            if (not i == 0) and (prev_time + h < t):
                u = prev_control
        else:
            h = t - prev_time  
        sim[i] = x
        x = motion_model(x, u, t, h)
        prev_time = t
        prev_control = u
    return tspan, sim

def extended_kalman(u_traj,
                    z_traj,
                    landmarks,
                    subject_dict,
                    motion_model,
                    measurement_model,
                    x0,
                    t0,
                    tf,
                    h,
                    Q,
                    R,
                    tspan=None, 
                    tsync='const'):
    """
    Function that executes a control trajectory given motion and measurement models,
    using the EKF for state estimation.

    Args:
        u_traj: Control trajectory
        z_traj: DataFrame containing measurements and corresponding timesteps
        landmarks: DataFrame of ground truth landmark data 
        subject_dict: Dictionary mapping measurement barcodes to subject numbers
        motion_model: Motion model function of a robot
        measurement_model: Measurement model function of a robot
        x0: Initial robot state
        t0: Simulation start time
        tf: Simulation end time
        h: Timestep
        R: Variance of Gaussian noise in motion model
        Q: Variance of Gaussian noise in measurement model
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
        
    """
    # Initialization
    z_traj = z_traj.to_numpy()
    if tspan is None: tspan = np.arange(start=t0, stop=tf, step=h)
    assert tspan.shape[0] == u_traj.shape[0]
    sim = np.zeros((len(tspan), len(x0)))
    prev_u_time = tspan[0] - h
    prev_control = u_traj[0]

    # Jacobian functions
    def G(mut, ut, h):
        return np.array([[1.0, 0.0, -ut[0]*h*np.sin(mut[2])],
                         [0.0, 1.0,  ut[0]*h*np.cos(mut[2])],
                         [0.0, 0.0,                     1.0]])
    def H(mut_bar, zt_bar, l, h):
        h_00 = (mut_bar[0] - l[0]) / zt_bar[0]
        h_01 = (mut_bar[1] - l[1]) / zt_bar[0]
        dx = l[0] - mut_bar[0]
        dy = l[1] - mut_bar[1]
        h_10 = dy / (dy**2 + dx**2)
        h_11 = -dx / (dy**2 + dx**2)
        return np.array([[h_00, h_01,  0.0],
                         [h_10, h_11, -1.0]])

    mut = x0
    sigt = np.diag(np.array([0.0001, 0.0001, 0.0001]))
    z_index = 0
    for (i, t), ut in zip(enumerate(tspan), u_traj):
        if tsync == 'const':
            # Control signals are not logged at a fixed timestep.
            # We can simulate at a fixed timestep, but send commands at proper times. In this implementation
            # the previous command is held onto, and used if it is still commanded at the current t.
            if (not i == 0) and (prev_u_time + h < t):
                ut = prev_control
        else:
            h = t - prev_u_time  

        # Make prediction
        sim[i] = mut
        mut_bar = motion_model(mut, ut, t, h)   # Mean prediction
        Gt = G(mut, ut, h)                      # Motion model Jacobian
        sigt_bar = (Gt @ sigt @ Gt.T) + R       # Variance prediction

        # Check if a new measurement has been made
        if z_index < z_traj.shape[0] and t >= z_traj[z_index, 0] and subject_dict[z_traj[z_index, 1]] >= 6:  # Subjects 1-5 are other robots?
            # Get actual measurement
            zt_bar, l = measurement_model(mut_bar, landmarks, subject_dict[z_traj[z_index, 1]])
            zt = z_traj[z_index, 2:]

            # Compute Kalman gain
            Ht = H(mut_bar, zt_bar, l, h)
            Kt = (sigt_bar @ Ht.T) @ np.linalg.inv((Ht @ sigt_bar @ Ht.T) + Q)
        
            # Update prediction with measurement
            z_diff = zt - zt_bar
            z_diff[1] = normalize_angle(z_diff[1])
            mut = mut_bar + Kt @ (z_diff)
            sigt = (np.eye(3) - Kt @ Ht) @ sigt_bar

            z_index += 1 
        else:
            # No measurement to update, save raw prediction
            mut = mut_bar
            sigt = sigt_bar

        mut[2] = normalize_angle(mut[2])
        prev_u_time = t
        prev_control = ut
    return tspan, sim

def unscented_kalman(u_traj,
                     z_traj,
                     landmarks,
                     subject_dict,
                     motion_model,  
                     measurement_model,
                     x0,
                     t0,
                     tf,
                     h,
                     Q,
                     R,
                     alp=1e-5,
                     k=0.0,
                     beta=2.0,
                     tspan=None, 
                     tsync='const'):
    """
    Function that executes a control trajectory given motion and measurement models,
    using the UKF for state estimation.

    Args:
        u_traj: Control trajectory
        z_traj: Measurement trajectory
        landmarks: Ground truth landmark data
        subject_dict: Dictionary mapping measurement barcodes to subject numbers
        motion_model: Motion model function of a robot
        measurement_model: Measurement model function of a robot
        x0: Initial robot state
        t0: Simulation start time
        tf: Simulation end time
        h: Timestep
        R: Variance of Gaussian noise in motion model
        Q: Variance of Gaussian noise in measurement model
        alp: Hyperparam controlling the spread of sigma points
        k: Scaling parameter
        beta: Hyperparam encoding prior knowledge of distribution
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
    """
    return
