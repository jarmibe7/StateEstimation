"""
run.py

Filters for HW0 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 09/19/2025
""" 
import numpy as np

def normalize_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

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
    sigt = np.diag(np.array([0.001, 0.001, 0.001]))
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
        updated = False
        if z_index < z_traj.shape[0] and t >= z_traj[z_index, 0]:  
            init_z_time = z_traj[z_index, 0]

            # Update based on all measurements at this timestep
            while z_index < z_traj.shape[0] and z_traj[z_index, 0] == init_z_time:
                if not subject_dict[z_traj[z_index, 1]] >= 6:  # Subjects 1-5 are other robots?
                    z_index += 1
                    continue
                # Get actual measurement
                zt = z_traj[z_index, 2:]
                zt_bar, l = measurement_model(mut_bar, landmarks, subject_dict[z_traj[z_index, 1]])

                # Compute Kalman gain
                Ht = H(mut_bar, zt_bar, l, h)
                Kt = (sigt_bar @ Ht.T) @ np.linalg.inv((Ht @ sigt_bar @ Ht.T) + Q)
            
                # Update prediction with measurement
                z_diff = zt - zt_bar
                z_diff[1] = normalize_angle(z_diff[1])
                mut = mut_bar + Kt @ (z_diff)
                sigt = (np.eye(3) - Kt @ Ht) @ sigt_bar

                updated = True
                z_index += 1 
        # No measurement to update, save raw prediction
        if not updated:
            mut = mut_bar
            sigt = sigt_bar
            new_meas = False

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
                     alp=1e-3,
                     k=0.0,
                     beta=2.0,
                     aug=False,
                     tspan=None, 
                     tsync='const'):
    """
    Function that executes a control trajectory given motion and measurement models,
    using the UKF for state estimation.
    
    Followed this tutorial by James Han on YouTube:
    https://www.youtube.com/watch?v=c_6WDC66aVk

    For augmented, also used this paper:
    chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://skoge.folk.ntnu.no/prost/proceedings/acc05/PDFs/Papers/0724_FrB02_6.pdf

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
        aug: Whether to use augmented mean and variance, and pass noise through nonlinear models
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

    # Calculate scaling and propogation parameters
    n_states = x0.shape[0]
    if aug: n = 2*n_states
    else: n = n_states
    lam = (alp**2)*(n + k) - n
    w0_m = lam / (n + lam)
    w0_c = w0_m + (1 - alp**2 + beta)
    wi_m = 1 / (2*(n + lam))
    wi_c = wi_m
    weights_m = np.full((2*n + 1), wi_m)  
    weights_c = np.full((2*n + 1), wi_c)
    weights_m[0] = w0_m     # First weight is special
    weights_c[0] = w0_c

    def sample_sigma(mut, sigt):
        """
        Sample sigma points from a given Gaussian
        """
        Xt = np.zeros(((2*n + 1), n))
        S = np.linalg.cholesky((n + lam)*sigt)  # Matrix square root
        Xt[0] = mut
        for j in range(n):
            Sj = S[:, j]
            Xt[j + 1] = mut + Sj
            Xt[j + n + 1] = mut - Sj

        return Xt

    mut = x0
    sigt = 0.01*np.eye(n_states)
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
        sim[i] = mut

        # Make augmented state vector and covariance
        if aug:
            mut_aug = np.hstack([mut, np.zeros(n_states)])
            sigt_aug = np.block([
                [sigt, np.zeros((n_states,n_states))],
                [np.zeros((n_states,n_states)),    R]
            ])
        else:
            mut_aug = mut
            sigt_aug = sigt

        # Sample sigma points
        Xt = sample_sigma(mut_aug, sigt_aug)

        # Pass sigma points through motion model
        Yt = np.array([motion_model(xj[:n_states], ut, t, h, w=xj[n_states:]) for xj in Xt])

        # Compute Gaussian statistics from transformed points
        mut_bar = weights_m @ Yt
        sigt_bar = np.zeros((n_states,n_states))
        if not aug: sigt_bar += R   # If not using augmented add R for noise estimation
        for j, yj in enumerate(Yt):
            sigt_bar += weights_c[j]*np.outer(yj - mut_bar, yj - mut_bar)

        # Check if a new measurement has been made
        updated = False
        if z_index < z_traj.shape[0] and t >= z_traj[z_index, 0]:  
            init_z_time = z_traj[z_index, 0]

            # Update based on all measurements at this timestep
            while z_index < z_traj.shape[0] and z_traj[z_index, 0] == init_z_time:
                if not subject_dict[z_traj[z_index, 1]] >= 6:  # Subjects 1-5 are other robots?
                    z_index += 1
                    continue
                # Get actual measurement
                subj = subject_dict[z_traj[z_index, 1]]
                zt = z_traj[z_index, 2:]

                # Make augmented state vector and covariance from prediction
                if aug:
                    mut_bar_aug = np.hstack([mut_bar, np.zeros(n_states)])
                    sigt_bar_aug = np.block([
                        [sigt_bar, np.zeros((n_states,n_states))],
                        [np.zeros((n_states,n_states)),        Q]
                    ])
                else:
                    mut_bar_aug = mut_bar
                    sigt_bar_aug = sigt_bar
            
                # Redraw sigma points from prediction
                Xt_bar = sample_sigma(mut_bar_aug, sigt_bar_aug)

                # Pass new sigma points through measurment model
                Zt = np.zeros((Xt_bar.shape[0], zt.shape[0]))
                for j, xj_bar in enumerate(Xt_bar):
                    zj_bar, l = measurement_model(xj_bar[:3], landmarks, subj, w=xj_bar[3:])
                    Zt[j] = zj_bar

                # Compute Gaussian statistics from transformed points
                zt_bar = weights_m @ Zt
                sigt_xz = np.zeros((Yt.shape[1], Zt.shape[1]))
                sigt_zz = np.zeros((Zt.shape[1], Zt.shape[1]))
                if not aug: sigt_zz += Q    # Approximate noise if not using augmented
                for j in range(Zt.shape[0]):
                    zj = Zt[j]
                    yj = Yt[j]
                    sigt_xz += weights_c[j]*np.outer(yj - mut_bar, zj - zt_bar)
                    sigt_zz += weights_c[j]*np.outer(zj - zt_bar, zj - zt_bar)

                # Incorporate measurement Gaussian into prediction to get posterior
                Kt = sigt_xz @ np.linalg.inv(sigt_zz)   # Kalman gain
                z_diff = zt - zt_bar
                z_diff[1] = normalize_angle(z_diff[1])
                mut = mut_bar + Kt @ (z_diff)
                sigt = sigt_bar - Kt @ sigt_zz @ Kt.T
                # sigt = sigt_bar - Kt @ sigt_xz.T
                
                updated = True
                z_index += 1
        
        # No measurement to update, save raw prediction
        if not updated:
            mut = mut_bar
            sigt = sigt_bar

        mut[2] = normalize_angle(mut[2])
        prev_u_time = t
        prev_control = ut
    return tspan, sim
