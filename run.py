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
import json

from filters import dead_reckon, extended_kalman, unscented_kalman, particle

PLOT_PATH = os.path.join(__file__, "../plots")
DATA_PATH = os.path.join(__file__, "../data")
METRICS_PATH = os.path.join(__file__, "../metrics")

#
# --- Evaluation Metrics ---
#
def t_match(traj, num_samples):
    """
    Resample a trajectory to have a certain number of samples
    """
    old_path_idx = np.linspace(0, 1, traj.shape[0])
    new_path_idx = np.linspace(0, 1, num_samples)

    traj_resamp = np.column_stack([
        np.interp(new_path_idx, old_path_idx, traj[:, i]) for i in range(traj.shape[1])
    ])

    return traj_resamp


# def mape(predicted, actual, angle=False):
#     """
#     Given two numpy arrays of the same length, compute Mean Average Predicted Error
#     between them.
#     """
#     assert type(actual) == np.ndarray, 'Parameters must be type np.ndarray'
#     assert type(predicted) == np.ndarray, 'Parameters must be type np.ndarray'
#     assert actual.shape == predicted.shape, 'Arrays must be of same shape'
#     if angle: return np.mean((np.abs(np.unwrap(actual - predicted))) / np.unwrap(actual)) * 100
#     else: return np.mean(np.abs((actual - predicted) / actual)) * 100
def rmse(predicted, actual, angle=False):
    """
    Given two 1D numpy arrays of the same length, compute Root Mean Squared Error
    between them.
    """
    if angle: error = np.unwrap(actual - predicted)
    else: error = np.linalg.norm(actual - predicted)
    return np.sqrt(error)

def compute_traj_statistics(predicted, actual):
    """
    Given a trajectory, compute various statistics about it from a ground truth.
    """
    stats = {}
    stats['rmse_x'] = rmse(predicted[:, 0], actual[:, 0])
    stats['rmse_y'] = rmse(predicted[:, 1], actual[:, 1])
    stats['rmse_theta'] = rmse(predicted[:, 2], actual[:, 2])
    stats['corr_x'] = np.corrcoef(predicted[:, 0], actual[:, 0])[0, 1]
    stats['corr_y'] = np.corrcoef(predicted[:, 1], actual[:, 1])[0, 1]
    stats['corr_theta'] = np.corrcoef(predicted[:, 2], actual[:, 2])[0, 1]

    return stats


#
# --- Models ---
#
def measurement_model(xt, landmarks_truth, subj, w=None):
    """
    Range-bearing measurement model

    Args:
        xt: State at current timestep
        landmarks_truth: Ground truth landmark 
        subj: Landmark subject number
        noise: Whether to incorporate noise from UKF into model
    """
    if w is None or w.shape[0] == 0:
        w = np.zeros((2,)) # Dummy noise vector

    # Get ground truth of landmark
    l = np.array(landmarks_truth.loc[landmarks_truth["subject"] == subj]).flatten()[1:]

    # Get average measuremnts
    mu_range = np.linalg.norm(xt[0:2] - l[0:2])
    mu_bearing = np.arctan2(l[1] - xt[1], l[0] - xt[0]) - xt[2]

    zt_bar = np.array([mu_range, mu_bearing]) + w[:2]

    return zt_bar, l

def motion_model(x, u, t, h, w=None):
    """
    A motion model that leverages RK4 integration for improved integration

    Args:
        x: Previous state
        u: Previous control
        t: Current time
        h: Timestep
        w: Noise vector to incorporate into model
    """
    if w is None or w.shape[0] == 0:
        w = np.zeros(x.shape)   # Dummy noise vector

    def f(x, t, u):
        """
        Nonlinear dynamics for planar wheeled robot.

        Args:
            x: State at current timestep
            t: Current timestep
            u_func: Control signal at current timestep
        """
        u_mult = np.array([u[0], u[0], u[1]]) + w
        xdot = np.array([np.cos(x[2]), np.sin(x[2]), 1])
        dyn = xdot * u_mult
        return dyn
    
    k1 = f(x, t, u)
    k2 = f(x + h*k1/2.0, t + h/2.0, u)
    k3 = f(x + h*k2/2.0, t + h/2.0, u)
    k4 = f(x + h*k3, t + h, u)

    return x + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)

#
# --- Simulation and Plotting ---
#
def gen_u_traj_test(h):
    """
    Generate test control trajectory for q2
    """
    seg_length = int(1.0 / h)
    straight = np.tile(np.array([0.5, 0.0]), (seg_length, 1))
    right_turn = np.tile(np.array([0.0, -1.0/(2.0*np.pi)]), (seg_length, 1))
    left_turn = -right_turn

    return np.vstack([straight, right_turn, straight, left_turn, straight])

def plot_wheeled_robot(trajectories, title, filename, plot_dr=True):
    """
    Plot the trajectory followed by a wheeled robot in the X-Y Plane
    """
    # Plot path
    if plot_dr: length = 8
    else: length = 5
    fig, ax = plt.subplots(1, 1, figsize=(length,4), tight_layout=True)
    for traj, label, display_robot, color, offset in trajectories:    # Plot multiple trajectories
        ax.plot(traj[:, 0], traj[:, 1], label=label, color=color)

        if display_robot:
            for i, xt in enumerate(traj):
                if i % (traj.shape[0] // 25) == 0 or i == 0:
                    # Plot robot location and heading
                    ax.plot(xt[0], xt[1], linestyle='', marker='h', markersize=8, color=color)
                    th_x = xt[0] + offset*np.cos(xt[2])
                    th_y = xt[1] + offset*np.sin(xt[2])
                    ax.plot(th_x, th_y, linestyle='', marker='*', markersize=5, color=color)
        
    plt.title(title)
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
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
    tspan, x_traj = dead_reckon(u_traj, motion_model, x0, t0, tf, h, tsync='const')

    _ = plot_wheeled_robot([(x_traj, 'Robot Trajectory', True, 'blue', 0.02)], "Wheeled Robot Trajectory in X-Y Plane (Q2)", "q2.png")
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
    tspan, x_traj = dead_reckon(u_traj, motion_model, x0, t0, tf, h, tspan=u_df['time'], tsync='var')

    # Plotting
    trajectories = [
        (x_traj, 'Dead-Reckoned', True, 'blue', 0.2),
        (np.array(ground_truth.iloc[:, 1:]), 'Ground Truth', True, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories, "Dead-Reckoned and Ground Truth Trajectories (Q3)", "q3.png")
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
        measured_landmarks.append(zt)
        true_landmarks.append(l)

    # Plot measurements and ground truth
    print("Done\n")
    fig, ax = plt.subplots(1, 1, figsize=(10,4), tight_layout=True)
    colors = ['red', 'green', 'blue']
    for zt, l, (xt, subj), c in zip(measured_landmarks, true_landmarks, test_landmarks, colors):
        ax.plot(l[0], l[1], linestyle='', marker='x', markersize=10, color=c, label=f'True Landmark {subj}')

        # Plot robot location and heading
        ax.plot(xt[0], xt[1], linestyle='', marker='h', markersize=8, color=c, label=f'Robot (Landmark {subj})')
        th_x = xt[0] + 0.07*np.cos(xt[2])
        th_y = xt[1] + 0.07*np.sin(xt[2])
        ax.plot(th_x, th_y, linestyle='', marker='*', markersize=5, color=c, label=f'Robot Heading (Landmark {subj})')

        # Determine landmark position from range and bearing
        zt_x = xt[0] + zt[0]*np.cos(zt[1])
        zt_y = xt[1] + zt[0]*np.sin(zt[1])
        ax.plot(zt_x, zt_y, linestyle='', marker='o', markersize=8, color=c, label=f'Measured Landmark {subj}')

        print(
            f'Landmark {subj:2d} | '
            f'Raw Measurements: ({zt[0]:.4f}, {zt[1]:.4f}) '
            f'Predictions: ({zt_x:.4f}, {zt_y:.4f}) '
            f'Error: ({(np.abs(zt_x) - np.abs(l[0])):.4f}, {(np.abs(zt_y) - np.abs(l[1])):.4f})'
        )
    # Move legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Comparison of Measured and Ground Truth Landmark Positions')
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    # plt.legend()
    fig_path = os.path.join(PLOT_PATH, 'q6.png')
    plt.savefig(fig_path)
        
def q8a():
    print("Running question 8a...", end="", flush=True)
    # Read data
    landmarks_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks = pd.read_csv(landmarks_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    measurement_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Measurement.dat')
    z_df = pd.read_csv(measurement_data_path, sep=r"\s+", comment="#", header=None, names=["time", "barcode", "range", "bearing"])
    barcodes_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Barcodes.dat')
    barcodes_df = pd.read_csv(barcodes_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "barcodes"])
    subject_dict = barcodes_df.set_index("barcodes")["subject"].to_dict()   # Map barcodes to subject number

    # Simulation conditions and run simulation
    x0 = np.zeros((3,))
    t0 = 0.0
    tf = 5.0
    h = 0.01
    u_traj = gen_u_traj_test(h) # Generate control trajectory
    tspan, x_traj_dr = dead_reckon(u_traj, motion_model, x0, t0, tf, h, tsync='const')
    R = np.diag(np.array([0.0001, 0.0001, 0.0001])) # Low noise
    q_val = 0.0001
    Q = np.diag(np.array([q_val, q_val]))
    tspan_ukf, x_traj_ukf_low = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tsync='const')
    R = np.diag(np.array([0.001, 0.001, 0.001])) # Medium noise
    q_val = 0.001
    Q = np.diag(np.array([q_val, q_val]))
    tspan_ukf, x_traj_ukf_med = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tsync='const')
    R = np.diag(np.array([0.01, 0.01, 0.01])) # Higher noise
    q_val = 0.01
    Q = np.diag(np.array([q_val, q_val]))
    tspan_ukf, x_traj_ukf_high = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tsync='const')

    disp_bots = True
    trajectories = [
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.02),
        (x_traj_ukf_low, 'UKF (Low Noise)', disp_bots, 'red', 0.02),
        (x_traj_ukf_med, 'UKF (Med. Noise)', disp_bots, 'green', 0.02),
        (x_traj_ukf_high, 'UKF (High Noise)', disp_bots, 'purple', 0.02),
    ]
    _ = plot_wheeled_robot(trajectories, "Dead-Reckoned and UKF Comparison for Artificial Control Trajectory (Q8a)", "q8a.png")
    print("Done\n")

def q8b_9():
    print("Running question 8b and 9...", end="", flush=True)
    # Read data
    controls_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Control.dat')
    u_df = pd.read_csv(controls_data_path, sep=r"\s+", comment="#", header=None, names=["time", "vel", "omega"])
    landmarks_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks = pd.read_csv(landmarks_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    measurement_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Measurement.dat')
    z_df = pd.read_csv(measurement_data_path, sep=r"\s+", comment="#", header=None, names=["time", "barcode", "range", "bearing"])
    barcodes_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Barcodes.dat')
    barcodes_df = pd.read_csv(barcodes_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "barcodes"])
    subject_dict = barcodes_df.set_index("barcodes")["subject"].to_dict()   # Map barcodes to subject number
    
    # Simulation conditions and run simulation
    x0 = np.array(ground_truth.iloc[0][1:])
    t0 = u_df['time'].iloc[0]
    tf = u_df['time'].iloc[-1]
    h = 1/67.0  # Odometry logged at 67 Hz
    u_traj = np.array(u_df.iloc[:, 1:])

    R = np.diag(np.array([0.000001, 0.000001, 0.000001]))
    q_val = 0.000001
    Q = np.diag(np.array([q_val, q_val]))   # No noise
    tspan_dr, x_traj_dr = dead_reckon(u_traj, motion_model, x0, t0, tf, h, Q, tspan=u_df['time'], tsync='var')
    tspan_ukf_none, x_traj_ukf_none = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')

    R = np.diag(np.array([0.0001, 0.0001, 0.0001]))
    q_val = 0.0001
    Q = np.diag(np.array([q_val, q_val]))   # Low noise
    tspan_dr, x_traj_dr = dead_reckon(u_traj, motion_model, x0, t0, tf, h, Q, tspan=u_df['time'], tsync='var')
    tspan_ukf_low, x_traj_ukf_low = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    
    R = np.diag(np.array([0.01, 0.01, 0.01]))   # Moderate noise
    q_val = 0.01
    Q = np.diag(np.array([q_val, q_val]))
    x_traj_gt = np.array(ground_truth.iloc[:, 1:])
    tspan_ukf, x_traj_ukf = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    
    R = np.diag(np.array([0.1, 0.1, 0.1]))   # High noise
    q_val = 0.1
    Q = np.diag(np.array([q_val, q_val]))
    x_traj_gt = np.array(ground_truth.iloc[:, 1:])
    tspan_ukf, x_traj_ukf_high = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')

    # Plotting
    disp_bots = False
    trajectories_8b = [     # No noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_none, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_8b, "Dead-Reckoned vs. UKF Comparison with No Noise (Q9)", "q9_none.png")
    trajectories_8b = [     # Low noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_low, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_8b, "Dead-Reckoned vs. UKF Comparison with Low Noise (Q9)", "q9_low.png")
    trajectories_8b = [     # Moderate Noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_8b, "Dead-Reckoned vs. UKF Comparison (Q8b)", "q8b.png")
    trajectories_8b = [     # High noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_high, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_8b, "Dead-Reckoned vs. UKF Comparison with High Noise (Q9)", "q9_high.png")

    # Compute statistics
    num_samples = x_traj_gt.shape[0] + (x_traj_ukf.shape[0] - x_traj_gt.shape[0])//2
    traj_dr_resamp = t_match(x_traj_dr, num_samples)
    traj_ukf_none_resamp = t_match(x_traj_ukf_none, num_samples)
    traj_ukf_low_resamp = t_match(x_traj_ukf_low, num_samples)
    traj_ukf_resamp = t_match(x_traj_ukf, num_samples)
    traj_ukf_high_resamp = t_match(x_traj_ukf_high, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_dr = compute_traj_statistics(traj_dr_resamp, traj_gt_resamp)
    stats_ukf_none = compute_traj_statistics(traj_ukf_none_resamp, traj_gt_resamp)
    stats_ukf_low = compute_traj_statistics(traj_ukf_low_resamp, traj_gt_resamp)
    stats_ukf = compute_traj_statistics(traj_ukf_resamp, traj_gt_resamp)
    stats_ukf_high = compute_traj_statistics(traj_ukf_high_resamp, traj_gt_resamp)

    metrics_dict = {'dead_reckoned': stats_dr, 'ukf_none': stats_ukf_none, 
                    'ukf_low': stats_ukf_low, 'ukf': stats_ukf, 'ukf_high': stats_ukf_high}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)
    print("Done\n")

def aug():
    print("Running augmented comparison...", end="", flush=True)
    # Read data
    controls_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Control.dat')
    u_df = pd.read_csv(controls_data_path, sep=r"\s+", comment="#", header=None, names=["time", "vel", "omega"])
    landmarks_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks = pd.read_csv(landmarks_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    measurement_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Measurement.dat')
    z_df = pd.read_csv(measurement_data_path, sep=r"\s+", comment="#", header=None, names=["time", "barcode", "range", "bearing"])
    barcodes_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Barcodes.dat')
    barcodes_df = pd.read_csv(barcodes_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "barcodes"])
    subject_dict = barcodes_df.set_index("barcodes")["subject"].to_dict()   # Map barcodes to subject number

    # Simulation conditions and run simulation
    x0 = np.array(ground_truth.iloc[0][1:])
    t0 = u_df['time'].iloc[0]
    tf = u_df['time'].iloc[-1]
    h = 1/67.0  # Odometry logged at 67 Hz
    u_traj = np.array(u_df.iloc[:, 1:])
    R = np.diag(np.array([0.01, 0.01, 0.01]))
    q_val = 0.1
    Q = np.diag(np.array([q_val, q_val]))
        
    tspan_ukf, x_traj_ukf = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Regular UKF
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    

    R = np.diag(np.array([0.000001, 0.000001, 0.000001]))
    q_val = 0.000001
    Q = np.diag(np.array([q_val, q_val, q_val]))   # No noise aug
    tspan_dr, x_traj_dr = dead_reckon(u_traj, motion_model, x0, t0, tf, h, Q, tspan=u_df['time'], tsync='var')
    tspan_ukf_none, x_traj_ukf_aug_none = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')

    R = np.diag(np.array([0.0001, 0.0001, 0.0001]))
    q_val = 0.0001
    Q = np.diag(np.array([q_val, q_val, q_val]))   # Low noise aug
    tspan_dr, x_traj_dr = dead_reckon(u_traj, motion_model, x0, t0, tf, h, Q, tspan=u_df['time'], tsync='var')
    tspan_ukf_low, x_traj_ukf_aug_low = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')
    
    R = np.diag(np.array([0.01, 0.01, 0.01]))   # Moderate noise aug
    q_val = 0.01
    Q = np.diag(np.array([q_val, q_val, q_val]))
    tspan_ukf, x_traj_ukf_aug = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')
    
    R = np.diag(np.array([0.1, 0.1, 0.1]))   # High noise aug
    q_val = 0.1
    Q = np.diag(np.array([q_val, q_val, q_val]))
    x_traj_gt = np.array(ground_truth.iloc[:, 1:])
    tspan_ukf, x_traj_ukf_aug_high = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model, 
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')

    # Plotting
    disp_bots = False
    trajectories_comp = [     # No noise
        (x_traj_ukf, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_ukf_aug_none, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_comp, "Regular UKF vs. Augmented UKF Comparison", "aug_reg.png", plot_dr=False)
    trajectories_no = [     # No noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_aug_none, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_no, "Dead-Reckoned vs. Augmented UKF Comparison with No Noise", "aug_none.png")
    trajectories_low = [     # Low noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_aug_low, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_low, "Dead-Reckoned vs. Augmented UKF Comparison with Low Noise", "aug_low.png")
    trajectories = [     # Moderate Noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_aug, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories, "Dead-Reckoned vs. Augmented UKF Comparison", "aug.png")
    trajectories_high = [     # High noise
        (x_traj_dr, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_aug_high, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_high, "Dead-Reckoned vs. Augmented UKF Comparison with High Noise", "aug_high.png")

    # Compute statistics
    num_samples = x_traj_gt.shape[0] + (x_traj_ukf.shape[0] - x_traj_gt.shape[0])//2
    traj_dr_resamp = t_match(x_traj_dr, num_samples)
    traj_ukf_aug_none_resamp = t_match(x_traj_ukf_aug_none, num_samples)
    traj_ukf_aug_low_resamp = t_match(x_traj_ukf_aug_low, num_samples)
    traj_ukf_aug_resamp = t_match(x_traj_ukf_aug, num_samples)
    traj_ukf_aug_high_resamp = t_match(x_traj_ukf_aug_high, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_dr = compute_traj_statistics(traj_dr_resamp, traj_gt_resamp)
    stats_ukf_aug_none = compute_traj_statistics(traj_ukf_aug_none_resamp, traj_gt_resamp)
    stats_ukf_aug_low = compute_traj_statistics(traj_ukf_aug_low_resamp, traj_gt_resamp)
    stats_ukf_aug = compute_traj_statistics(traj_ukf_aug_resamp, traj_gt_resamp)
    stats_ukf_aug_high = compute_traj_statistics(traj_ukf_aug_high_resamp, traj_gt_resamp)

    metrics_dict = {'dead_reckoned': stats_dr, 'ukf_aug_none': stats_ukf_aug_none, 
                    'ukf_aug_low': stats_ukf_aug_low, 'ukf_aug': stats_ukf_aug, 'ukf_aug_high': stats_ukf_aug_high}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)
    print("Done\n")

def study_comp():
    print("Running study comparison...", end="", flush=True)
    # Read data
    controls_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Control.dat')
    u_df = pd.read_csv(controls_data_path, sep=r"\s+", comment="#", header=None, names=["time", "vel", "omega"])
    landmarks_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks = pd.read_csv(landmarks_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    measurement_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Measurement.dat')
    z_df = pd.read_csv(measurement_data_path, sep=r"\s+", comment="#", header=None, names=["time", "barcode", "range", "bearing"])
    barcodes_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Barcodes.dat')
    barcodes_df = pd.read_csv(barcodes_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "barcodes"])
    subject_dict = barcodes_df.set_index("barcodes")["subject"].to_dict()   # Map barcodes to subject number

    # Simulation conditions and run simulation
    x0 = np.array(ground_truth.iloc[0][1:])
    t0 = u_df['time'].iloc[0]
    tf = u_df['time'].iloc[-1]
    h = 1/67.0  # Odometry logged at 67 Hz
    u_traj = np.array(u_df.iloc[:, 1:])
    p = 1.0
    R = np.diag(np.array([p, p, p]))
    Q = np.diag(np.array([p, p]))
        
    tspan_ukf, x_traj_ukf_slow = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Regular UKF p=1.0
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    
    p = 10.0
    R = np.diag(np.array([p, p, p]))
    Q = np.diag(np.array([p, p]))
        
    tspan_ukf, x_traj_ukf_shigh = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Regular UKF p=10.0
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    
    p = 1.0
    R = np.diag(np.array([p, p, p]))
    Q = np.diag(np.array([p, p, p]))
        
    tspan_ukf, x_traj_ukf_aug_slow = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Augmented UKF p=1.0
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')
    
    p = 10.0
    R = np.diag(np.array([p, p, p]))
    Q = np.diag(np.array([p, p, p]))
        
    tspan_ukf, x_traj_ukf_aug_shigh = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Augmented UKF p=10.0
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')
    
    x_traj_gt = np.array(ground_truth.iloc[:, 1:])

    # Plotting
    disp_bots = False
    trajectories_slow = [     # No noise
        (x_traj_ukf_slow, 'UKF', disp_bots, 'red', 0.2),
        (x_traj_ukf_aug_slow, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_slow, "Regular UKF vs. Augmented UKF Comparison with p=1.0", "slow.png", plot_dr=False)
    trajectories_shigh = [     # No noise
        (x_traj_ukf_shigh, 'Dead-Reckoned', disp_bots, 'blue', 0.2),
        (x_traj_ukf_aug_shigh, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_shigh, "Regular UKF vs. Augmented UKF Comparison with p=10.0", "shigh.png", plot_dr=False)

    # Compute statistics
    num_samples = x_traj_gt.shape[0] + (x_traj_ukf_slow.shape[0] - x_traj_gt.shape[0])//2
    traj_ukf_slow_resamp = t_match(x_traj_ukf_slow, num_samples)
    traj_ukf_aug_slow_resamp = t_match(x_traj_ukf_aug_slow, num_samples)
    traj_ukf_shigh_resamp = t_match(x_traj_ukf_shigh, num_samples)
    traj_ukf_aug_shigh_resamp = t_match(x_traj_ukf_aug_shigh, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_ukf_slow = compute_traj_statistics(traj_ukf_slow_resamp, traj_gt_resamp)
    stats_ukf_aug_slow = compute_traj_statistics(traj_ukf_aug_slow_resamp, traj_gt_resamp)
    stats_ukf_shigh = compute_traj_statistics(traj_ukf_shigh_resamp, traj_gt_resamp)
    stats_ukf_aug_shigh = compute_traj_statistics(traj_ukf_aug_shigh_resamp, traj_gt_resamp)

    metrics_dict = {'ukf_slow': stats_ukf_slow, 'ukf_aug_slow': stats_ukf_aug_slow, 'ukf_shigh': stats_ukf_shigh, 'ukf_aug_shigh': stats_ukf_aug_shigh}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)
    print("Done\n")

def all_filters():
    print("Running all filters comparison...", end="", flush=True)
    # Read data
    controls_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Control.dat')
    u_df = pd.read_csv(controls_data_path, sep=r"\s+", comment="#", header=None, names=["time", "vel", "omega"])
    landmarks_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Landmark_Groundtruth.dat')
    landmarks = pd.read_csv(landmarks_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    truth_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Groundtruth.dat')
    ground_truth = pd.read_csv(truth_data_path, sep=r"\s+", comment="#", header=None, names=["time", "x", "y", "theta"])
    measurement_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Measurement.dat')
    z_df = pd.read_csv(measurement_data_path, sep=r"\s+", comment="#", header=None, names=["time", "barcode", "range", "bearing"])
    barcodes_data_path = os.path.join(DATA_PATH, 'ds0', 'ds0_Barcodes.dat')
    barcodes_df = pd.read_csv(barcodes_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "barcodes"])
    subject_dict = barcodes_df.set_index("barcodes")["subject"].to_dict()   # Map barcodes to subject number

    # Simulation conditions and run simulation
    x0 = np.array(ground_truth.iloc[0][1:])
    t0 = u_df['time'].iloc[0]
    tf = u_df['time'].iloc[-1]
    h = 1/67.0  # Odometry logged at 67 Hz
    u_traj = np.array(u_df.iloc[:, 1:])
    p = 0.01
    R = np.diag(np.array([p, p, p]))
    Q = np.diag(np.array([p, p]))

    tspan_ekf, x_traj_ekf = extended_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # EKF
                                             x0, t0, tf, h, Q, R, tspan=u_df['time'], tsync='var')

    tspan_ukf, x_traj_ukf = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Regular UKF
                                             x0, t0, tf, h, Q, R, aug=False, tspan=u_df['time'], tsync='var')
    
    tspan_part, x_traj_part = particle(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,    # Particle
                                     x0, t0, tf, h, Q, R, M=50, tspan=u_df['time'], tsync='var')

    Q = np.diag(np.array([p, p, p]))
    tspan_ukf_aug, x_traj_ukf_aug = unscented_kalman(u_traj, z_df, landmarks, subject_dict, motion_model, measurement_model,   # Augmented UKF
                                             x0, t0, tf, h, Q, R, aug=True, tspan=u_df['time'], tsync='var')
    
    x_traj_gt = np.array(ground_truth.iloc[:, 1:])
    
    disp_bots = False
    trajectories_ekf = [     # EKF
        (x_traj_ekf, 'EKF', disp_bots, 'grey', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_ekf, "EKF vs. Ground Truth", "all_ekf.png", plot_dr=False)
    trajectories_ukf = [     # UKF
        (x_traj_ukf, 'EKF', disp_bots, 'red', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_ukf, "Regular UKF vs. Ground Truth", "all_ukf.png", plot_dr=False)
    trajectories_part = [     # Particle Filter
        (x_traj_part, 'Particle Filter', disp_bots, 'black', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_part, "Particle Filter vs. Ground Truth", "all_part.png", plot_dr=False)
    trajectories_ukf_aug = [     # Augmented UKF
        (x_traj_ukf_aug, 'Augmented UKF', disp_bots, 'purple', 0.2),
        (x_traj_gt, 'Ground Truth', disp_bots, 'orange', 0.2)
    ]
    _ = plot_wheeled_robot(trajectories_ukf_aug, "Augmented UKF vs. Ground Truth", "all_ukf_aug.png", plot_dr=False)

    # Compute statistics
    num_samples = x_traj_gt.shape[0] + (x_traj_ukf_aug.shape[0] - x_traj_gt.shape[0])//2
    traj_ekf_resamp = t_match(x_traj_ekf, num_samples)
    traj_part_resamp = t_match(x_traj_part, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_ekf = compute_traj_statistics(traj_ekf_resamp, traj_gt_resamp)
    stats_part = compute_traj_statistics(traj_part_resamp, traj_gt_resamp)

    metrics_dict = {'ekf': stats_ekf, 'particle': stats_part}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)
    print("Done\n")
    

def main():
    print("*** STARTING ***\n")
    q2()
    q3()
    q6()
    q8a()
    q8b_9()
    # aug()
    # study_comp()
    # all_filters()
    
    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()