"""
run.py

Main script for HW0 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 09/19/2025
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

PLOT_PATH = os.path.join(__file__, "../plots")
DATA_PATH = os.path.join(__file__, "../data")

#
# --- Measurement ---
#
def measurement_model(xt, landmarks_truth, subj):
    """
    Measurement model based on range-bearing model on page 940 of
    Artifical Intelligence: A Modern Approach by Norvig et al.

    Args:
        xt: State at current timestep
        landmarks_truth: Ground truth landmark 
        subj: Landmark subject number
    """
    # Get ground truth of landmark
    l = np.array(landmarks_truth.loc[landmarks_truth["subject"] == subj]).flatten()[1:]

    # Get average measuremnts
    mu_range = np.linalg.norm(xt[0:2] - l[0:2])
    mu_bearing = np.arctan((l[1] - xt[1]) / (l[0] / xt[0])) - xt[2]

    # TODO: Sample measurement from distribution, but for now just return mean
    return np.array([mu_range, mu_bearing]), l

#
# --- Simulation Functions ---
#
def integrate_rk4(f, x0, t0, tf, h, u_traj, tspan=None, tsync='const'):
    """
    RK4 integration and simulator

    Args:
        f: Dynamics function
        x0: Initial state
        t0: Initial simulation time
        tf: Ending simulation time
        h: Timestep
        tspan: If given a preexisting time series, can optionally use that instead
        tsync: Time synchronization, either at constant or variable timestep - 'const' | 'var'
    """
    def rk4(f, x, u, t, h):
        k1 = f(x, t, u)
        k2 = f(x + h*k1/2.0, t + h/2.0, u)
        k3 = f(x + h*k2/2.0, t + h/2.0, u)
        k4 = f(x + h*k3, t + h, u)
        return x + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
    
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
        x = rk4(f, x, u, t, h)
        prev_time = t
        prev_control = u
    return tspan, sim

def gen_u_traj_test(h):
    """
    Generate test control trajectory for q2
    """
    seg_length = int(1.0 / h)
    straight = np.tile(np.array([0.5, 0.0]), (seg_length, 1))
    right_turn = np.tile(np.array([0.0, -1.0/(2.0*np.pi)]), (seg_length, 1))
    left_turn = -right_turn

    return np.vstack([straight, right_turn, straight, left_turn, straight])

def dynamics(x, t, u):
    """
    Motion model for planar wheeled robot.

    Args:
        x: State at current timestep
        t: Current timestep
        u_func: Control signal at current timestep
    """
    u_mult = np.array([u[0], u[0], u[1]])
    xdot = np.array([np.cos(x[2]), np.sin(x[2]), 1])
    return xdot * u_mult

def plot_wheeled_robot(trajectories, title, filename):
    """
    Plot the trajectory followed by a wheeled robot in the X-Y Plane
    """
    fig, ax = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)
    for traj, label in trajectories:    # Plot multiple trajectories
        ax.plot(traj[:, 0], traj[:, 1], label=label)
    plt.title(title)
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.legend()
    fig_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(fig_path)
    return fig

#
# --- Questions ---
#
def q2():
    print("Running question 2...", end="")
    # Simulation conditions and run simulation
    x0 = np.zeros((3,))
    t0 = 0.0
    tf = 5.0
    h = 0.01
    u_traj = gen_u_traj_test(h) # Generate control trajectory
    tspan, x_traj = integrate_rk4(dynamics, x0, t0, tf, h, u_traj, tsync='const')

    _ = plot_wheeled_robot([(x_traj, 'Robot Trajectory')], "Wheeled Robot Trajectory in X-Y Plane (Q2)", "q2.png")
    print("Done\n")

def q3():
    print("Running question 3...", end="")
    # Read controls and ground truth data
    controls_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Control.dat')
    u_df = pd.read_csv(controls_data_path, sep=r"\s+", comment="#", header=None, names=["time", "vel", "omega"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    
    # Simulation conditions and run simulation
    x0 = np.array(ground_truth.iloc[0][1:])
    t0 = u_df['time'].iloc[0]
    tf = u_df['time'].iloc[-1]
    h = 1/67.0  # Odometry logged at 67 Hz
    u_traj = np.array(u_df.iloc[:, 1:])
    tspan, x_traj = integrate_rk4(dynamics, x0, t0, tf, h, u_traj, tspan=u_df['time'], tsync='var')

    # Plotting
    trajectories = [
        (x_traj, 'Dead-Reckoned'),
        (np.array(ground_truth.iloc[:, 1:]), 'Ground Truth')
    ]
    _ = plot_wheeled_robot(trajectories, "Dead-Reckoned and Ground Truth Trajectories - ds0 (Q3)", "q3.png")
    print("Done\n")

def q6():
    print("Running question 6...", end="")
    # Read ground truth landmark data
    landmarks_truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks_truth = pd.read_csv(landmarks_truth_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])

    # Take test measurements
    test_landmarks = [
        (np.array([2, 3, 0]), 6), 
        (np.array([0, 3, 0]), 13), 
        (np.array([1, -2, 0]), 17)
    ]
    measured_landmarks = []
    true_landmarks = []
    for xt, subj in test_landmarks:
        zt, l = measurement_model(xt, landmarks_truth, subj)
        print(f'Measured Landmark {subj}: ({zt[0]}, {zt[1]})')
        measured_landmarks.append(zt)
        true_landmarks.append(l)

    # Plot measurements and ground truth
    fig, ax = plt.subplots(1, 1, figsize=(10,4), tight_layout=True)
    colors = ['red', 'green', 'blue']
    for zt, l, (xt, subj), c in zip(measured_landmarks, true_landmarks, test_landmarks, colors):
        ax.plot(l[0], l[1], linestyle='', marker='x', markersize=8, color=c, label=f'True Landmark {subj}')

        # Plot robot location and heading
        ax.plot(xt[0], xt[1], linestyle='', marker='h', markersize=8, color=c, label=f'Robot (Landmark {subj})')
        th_x = xt[0] + 0.07*np.cos(xt[2])
        th_y = xt[1] + 0.07*np.sin(xt[2])
        ax.plot(th_x, th_y, linestyle='', marker='*', markersize=5, color=c, label=f'Robot (Landmark {subj})')

        # Determine landmark position from range and bearing
        zt_x = xt[0] + zt[0]*np.cos(zt[1])
        zt_y = xt[1] + zt[0]*np.sin(zt[1])
        ax.plot(zt_x, zt_y, linestyle='', marker='o', markersize=8, color=c, label=f'Measured Landmark {subj}')

    # Move legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Comparison of Measured and Ground Truth Landmark Positions')
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    # plt.legend()
    fig_path = os.path.join(PLOT_PATH, 'q6.png')
    plt.savefig(fig_path)
    print("Done\n")

def main():
    print("*** STARTING ***\n")
    q2()
    q3()
    q6()
    
    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()