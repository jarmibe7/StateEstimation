"""
run.py

Filters for HW0 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 09/19/2025
""" 
import numpy as np
import pandas as pd
import os

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
    prev_control = u_traj
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
        x = motion_model(x, u, t, h, Q=Q)
        prev_time = t
        prev_control = u
    return tspan, sim

def extended_kalman(u_traj,
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
        motion_model: Motion model function of a robot
        measurement_model: Measurement model function of a robot
        x0: Initial robot state
        t0: Simulation start time
        tf: Simulation end time
        h: Timestep
        Q: Variance of Gaussian noise in motion model
        R: Variance of Gaussian noise in measurement model
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
        
    """
    # Initialization
    if tspan is None: tspan = np.arange(start=t0, stop=tf, step=h)
    assert tspan.shape[0] == u_traj.shape[0]
    x = x0
    sim = np.zeros((len(tspan), len(x0)))
    prev_time = tspan[0] - h
    prev_control = u_traj

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
        
        x = motion_model(x, u, t, h, Q=Q)
        prev_time = t
        prev_control = u
    return tspan, sim

def unscented_kalman(u_traj,
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
        motion_model: Motion model function of a robot
        measurement_model: Measurement model function of a robot
        x0: Initial robot state
        t0: Simulation start time
        tf: Simulation end time
        h: Timestep
        Q: Variance of Gaussian noise in motion model
        R: Variance of Gaussian noise in measurement model
        alp: Hyperparam controlling the spread of sigma points
        k: Scaling parameter
        beta: Hyperparam encoding prior knowledge of distribution
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
    """
    return
