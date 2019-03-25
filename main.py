import tqdm
import numpy as np
import os, copy, argparse
from scipy.linalg import expm
import matplotlib.pyplot as plt
from utils import *

parser = argparse.ArgumentParser(description='Visual-Inertial SLAM -- ECE 276A Project #3')
parser.add_argument('-d', dest='dataset', type=str, default='0020', help='dataset name, eg. 0027')
parser.add_argument('-f', dest='FUSED', action='store_true', help='update landmark and pose in fusion')
parser.add_argument('-c', dest='CHECK', action='store_true', help='check if landmarks are static')
parser.add_argument('-w', dest='w', type=float, default=1e-7, help='noise level for imu prediction, default 1e-6')
parser.add_argument('-v', dest='v', type=float, default=1e2, help='noise level for landmarks observation, default 1e2')


if __name__ == '__main__':
    # Load data
    args = parser.parse_args()
    print('Configuration:')
    print('    ', args)

    filename = './data/' + args.dataset + '.npz'
    t, dt, features, linear_velocity, rotational_velocity, M, cam_T_imu = load_data(filename)
    imu_T_cam = np.linalg.inv(cam_T_imu)
    len_t = len(dt)
    imu_T_wld = np.zeros((4, 4, len_t))
    wld_T_imu = np.zeros((4, 4, len_t))
    Sigma_imu = np.zeros((6, 6, len_t))

    # initialization for EKF prediction
    imu_T_wld[:, :, 0] = np.eye(4)      # imu state: T_t
    wld_T_imu[:, :, 0] = np.eye(4)      # created for plotting
    Sigma_imu[:, :, 0] = np.eye(6)      # imu pose covariance matrix
    W = args.w*np.eye(6)                # imu prediction noise covariance


    # --- a prediction-only version, used only for comparing results
    for i in range(1, len_t):
        # IMU pose prediction via EKF
        u = np.hstack((linear_velocity[:, i], rotational_velocity[:, i])).squeeze()
        imu_T_wld[:, :, i] = expm(hat(-dt[i] * u)) @ imu_T_wld[:, :, i - 1]
        wld_T_imu[:, :, i] = np.linalg.inv(imu_T_wld[:, :, i])
        Exp = expm(-dt[i] * curly_hat(u))
        Sigma_imu[:, :, i] = Exp @ Sigma_imu[:, :, i - 1] @ Exp.T + dt[i] ** 2 * W
    # record for comparing results
    wld_T_imu_pre = copy.deepcopy(wld_T_imu)
    Sigma_imu_pre = copy.deepcopy(Sigma_imu)


    # initialization for EKF update
    if args.FUSED:
        M_ldk = features.shape[1]           # total number of landmarks (ldk) observed in the dataset
        mu_ldk0 = np.zeros((4, M_ldk))      # record initial state of each landmark, used for visualizing results
        mu = np.zeros((4, M_ldk + 4))       # fused state: 4 x M for landmarks, 4 x 4 for imu pose
        Sigma = np.eye(3*M_ldk + 6)         # fused observation covariance
        V = args.v * np.eye(4)              # landmarks observation noise covariance
        D = np.block([[np.eye(3)], [np.zeros((1, 3))]])

        # update landmarks using first observation
        indices = np.flatnonzero(features[0, :, 0] != -1)  # valid observation given current frame with size (N_t,)
        mu[:, indices] = mu_ldk0[:, indices] = \
            back_projection(features[:, indices, 0], M, imu_T_cam, wld_T_imu[:, :, 0])
    else:
        M_ldk = features.shape[1]           # total number of landmarks (ldk) observed in the dataset
        mu_ldk0 = np.zeros((4, M_ldk))      # record initial state of each landmark, used for visualizing results
        mu_ldk = np.zeros((4, M_ldk))       # landmarks state of current frame
        Sigma_ldk = np.eye(3*M_ldk)         # landmarks observation covariance
        V = args.v*np.eye(4)                # landmarks observation noise covariance
        D = np.block([[np.eye(3)], [np.zeros((1, 3))]])

        # update landmarks using first observation
        indices = np.flatnonzero(features[0, :, 0] != -1)   # valid observation given current frame with size (N_t,)
        mu_ldk[:, indices] = mu_ldk0[:, indices] = \
            back_projection(features[:, indices, 0], M, imu_T_cam, wld_T_imu[:, :, 0])


    # --- a complete Visual-Inertial SLAM algorithm including prediction and update
    for i in tqdm.trange(1, len_t, desc='VI-SLAM', unit='frame'):
        # IMU pose prediction via EKF
        u = np.hstack((linear_velocity[:, i], rotational_velocity[:, i])).squeeze()
        imu_T_wld[:, :, i] = expm(hat(-dt[i] * u)) @ imu_T_wld[:, :, i - 1]
        wld_T_imu[:, :, i] = np.linalg.inv(imu_T_wld[:, :, i])
        Exp = expm(-dt[i] * curly_hat(u))
        Sigma_imu[:, :, i] = Exp @ Sigma_imu[:, :, i - 1] @ Exp.T + dt[i] ** 2 * W

        if args.FUSED:
            # update covariance
            Sigma[-6:, -6:] = Sigma_imu[:, :, i]
            Sigma[:3*M_ldk, 3*M_ldk:] = Sigma[:3*M_ldk, 3*M_ldk:] @ Exp.T
            Sigma[3*M_ldk:, :3*M_ldk] = Sigma[:3*M_ldk, 3*M_ldk:].T

            # landmarks & pose update via EKF
            indices = np.flatnonzero(features[0, :, i] != -1)   # (N_t,)
            z_ldk = features[:, indices, i]                     # (4, N_t)
            N_t = len(indices)
            if N_t:
                # adding new landmarks if observed for the first time
                idx_new_ldk = indices[np.flatnonzero(mu_ldk0[-1, indices] == 0)]
                z_new_ldk = features[:, idx_new_ldk, i]
                mu[:, idx_new_ldk] = mu_ldk0[:, idx_new_ldk] = back_projection(z_new_ldk, M,
                                                                               imu_T_cam, wld_T_imu[:, :, i])
                # landmarks & pose update via EKF
                H = np.zeros((4*N_t, 3*M_ldk + 6))
                z_ldk_hat = M @ projection(cam_T_imu, imu_T_wld[:, :, i], mu[:, indices])
                if args.CHECK:
                    z_ldk_hat = check_static(z_ldk, z_ldk_hat)
                for j in range(N_t):
                    H[4*j:4*j + 4, 3*indices[j]:3*indices[j] + 3] = \
                        M @ d_proj(cam_T_imu @ imu_T_wld[:, :, i] @ mu[:, indices[j]].reshape(4, -1)) @ \
                        cam_T_imu @ imu_T_wld[:, :, i] @ D
                    H[4*j:4*j + 4, -6:] = M @ d_proj(cam_T_imu @ imu_T_wld[:, :, i] @ mu[:, indices[j]]) @ \
                                          cam_T_imu @ odot(imu_T_wld[:, :, i] @ mu[:, indices[j]].reshape(4, -1))
                K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + np.kron(np.eye(N_t), V))

                # update landmarks mean
                mu[:, :M_ldk] = (mu[:, :M_ldk].flatten('F') + np.kron(np.eye(M_ldk), D) @ K[:3*M_ldk, :] @
                                   (z_ldk - z_ldk_hat).flatten('F')).reshape(M_ldk, 4).T
                # update IMU pose mean
                mu[:, M_ldk:] = expm(hat(K[-6:, :] @ (z_ldk - z_ldk_hat).flatten('F'))) @ imu_T_wld[:, :, i]

                # update fused covariance
                Sigma = (np.eye(3*M_ldk + 6) - K @ H) @ Sigma

                # update record
                mu_ldk = mu[:, :M_ldk]
                imu_T_wld[:, :, i] = mu[:, M_ldk:]
                wld_T_imu[:, :, i] = np.linalg.inv(imu_T_wld[:, :, i])
                Sigma_imu[:, :, i] = Sigma[-6:, -6:]
        else:
            # landmarks & pose update via EKF
            indices = np.flatnonzero(features[0, :, i] != -1)  # (N_t,)
            z_ldk = features[:, indices, i]  # (4, N_t)
            N_t = len(indices)
            if N_t:
                # adding new landmarks if observed for the first time
                idx_new_ldk = indices[np.flatnonzero(mu_ldk0[-1, indices] == 0)]
                z_new_ldk = features[:, idx_new_ldk, i]
                mu_ldk[:, idx_new_ldk] = mu_ldk0[:, idx_new_ldk] = back_projection(z_new_ldk, M,
                                                                                   imu_T_cam, wld_T_imu[:, :, i])
                # landmarks update via EKF
                H = np.zeros((4 * N_t, 3 * M_ldk))
                z_ldk_hat = M @ projection(cam_T_imu, imu_T_wld[:, :, i], mu_ldk[:, indices])
                if args.CHECK:
                    z_ldk_hat = check_static(z_ldk, z_ldk_hat)
                for j in range(N_t):
                    H[4 * j:4 * j + 4, 3 * indices[j]:3 * indices[j] + 3] = \
                        M @ d_proj(cam_T_imu @ imu_T_wld[:, :, i] @ mu_ldk[:, indices[j]].reshape(4, -1)) @ \
                        cam_T_imu @ imu_T_wld[:, :, i] @ D
                K = Sigma_ldk @ H.T @ np.linalg.inv(H @ Sigma_ldk @ H.T + np.kron(np.eye(N_t), V))
                Sigma_ldk = (np.eye(3 * M_ldk) - K @ H) @ Sigma_ldk
                mu_ldk_updated = (mu_ldk.flatten('F') + np.kron(np.eye(M_ldk), D) @ K @
                                  (z_ldk - z_ldk_hat).flatten('F')).reshape(M_ldk, 4).T

                # IMU pose update via EKF
                H = np.zeros((4 * N_t, 6))
                for j in range(N_t):
                    H[4 * j:4 * j + 4] = M @ d_proj(cam_T_imu @ imu_T_wld[:, :, i] @ mu_ldk[:, indices[j]]) @ \
                                         cam_T_imu @ odot(imu_T_wld[:, :, i] @ mu_ldk[:, indices[j]].reshape(4, -1))
                K = Sigma_imu[:, :, i] @ H.T @ np.linalg.inv(H @ Sigma_imu[:, :, i] @ H.T + np.kron(np.eye(N_t), V))
                imu_T_wld[:, :, i] = expm(hat(K @ (z_ldk - z_ldk_hat).flatten('F'))) @ imu_T_wld[:, :, i]
                wld_T_imu[:, :, i] = np.linalg.inv(imu_T_wld[:, :, i])
                Sigma_imu[:, :, i] = (np.eye(6) - K @ H) @ Sigma_imu[:, :, i]

                # update landmarks state, this guarantees that we update the pose and landmarks at the same time
                mu_ldk = mu_ldk_updated

    # visualize Visual-Inertial SLAM results
    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    visualize_trajectory(axes[0], wld_T_imu_pre, wld_T_imu, title='trajectory & orientation')
    visualize_landmark(axes[1], wld_T_imu, mu_ldk0, mu_ldk, title='trajectory & landmarks')
    visualize_covariance(axes[2], t, Sigma_imu_pre, Sigma_imu, title='imu pose covariance')
    if args.FUSED and args.CHECK:
        plt.savefig(os.path.join(result_dir, args.dataset + '_f_c'), dpi=200, bbox_inches='tight', pad_inches=.3)
    elif args.FUSED and not args.CHECK:
        plt.savefig(os.path.join(result_dir, args.dataset + '_f_unc'), dpi=200, bbox_inches='tight', pad_inches=.3)
    elif not args.FUSED and args.CHECK:
        plt.savefig(os.path.join(result_dir, args.dataset + '_unf_c'), dpi=200, bbox_inches='tight', pad_inches=.3)
    else:
        plt.savefig(os.path.join(result_dir, args.dataset + '_unf_unc'), dpi=200, bbox_inches='tight', pad_inches=.3)
    plt.show(block=False)

    print('Done.')


