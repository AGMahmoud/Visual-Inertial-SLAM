import numpy as np
from transforms3d.euler import mat2euler


def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XXX_sync_KLT.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images,
            with shape 4*n*t, where n is number of features
        linear_velocity: IMU measurements in IMU frame
            with shape 3*t
        rotational_velocity: IMU measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            [fx  0 cx
              0 fy cy
              0  0  1]
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
            close to
            [ 0 -1  0 t1
              0  0 -1 t2
              1  0  0 t3
              0  0  0  1]
            with shape 4*4
    '''
    with np.load(file_name) as data:
        t = data["time_stamps"]         # time_stamps
        features = data["features"]     # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"]   # linear velocity measured in the body frame
        rotational_velocity = data["rotational_velocity"]   # rotational velocity measured in the body frame
        K = data["K"]       # intrinsic calibration matrix
        b = data["b"]       # baseline
        cam_T_imu = data["cam_T_imu"]       # Transformation from imu to camera frame

    # generate M based on K and b
    M = np.zeros((4, 4))
    M[:2, :3], M[2:, :3], M[2, 3] = K[:2, :], K[:2, :], -b * K[0, 0]
    # calculate dt
    t = t.squeeze()
    t -= t[0]
    dt = np.zeros(t.shape)
    dt[1:] = t[1:] - t[:-1]
    return t, dt, features, linear_velocity, rotational_velocity, M, cam_T_imu


def visualize_trajectory(ax, pose_pre, pose, title=None):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
        landmarks: 4*M matrix with homogeneous coords
    '''

    # show trajectory
    ax.plot(pose_pre[0, 3, :], pose_pre[1, 3, :], 'c-', label='inertial only')
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-', label='visual-inertial')
    # show orientation
    n_pose = pose_pre.shape[2]
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")
    select_ori_index = list(range(0, n_pose, int(n_pose / 50)))
    yaw_list = []
    for i in select_ori_index:
        _, _, yaw = mat2euler(pose[:3, :3, i])
        yaw_list.append(yaw)
    dx = np.cos(yaw_list)
    dy = np.sin(yaw_list)
    dx, dy = [dx, dy] / np.sqrt(dx ** 2 + dy ** 2)
    ax.quiver(pose[0, 3, select_ori_index], pose[1, 3, select_ori_index], dx, dy,
              color="b", units="xy", width=1)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()


def visualize_landmark(ax, pose, landmarks0, landmarks, title=None):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose,
                where N is the number of pose, and each
                4*4 matrix is in SE(3)
        landmarks: 4*M matrix with homogeneous coords
    '''
    ax.plot(landmarks0[0], landmarks0[1], 'c.', label='initial landmarks')
    ax.plot(landmarks[0], landmarks[1], 'k.', label='updated ones')
    ax.plot(pose[0, 3, :], pose[1, 3, :], 'r-')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()


def visualize_covariance(ax, t, Sigma_pre, Sigma, title=None):
    ax.plot(t, np.log(np.linalg.norm(Sigma_pre, axis=(0, 1))), label='prediction only')
    ax.plot(t, np.log(np.linalg.norm(Sigma, axis=(0, 1))), label='prediction & update')
    ax.set_xlabel('time / s')
    ax.set_ylabel('matrix norm / dB')
    ax.set_title(title)
    ax.grid()
    ax.legend(loc=1)


def hat(x):
    x = x.squeeze()
    if x.ndim != 1:
        raise ValueError('hat operation is only defined on 1-d array.')
    if len(x) == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    elif len(x) == 6:
        return np.block([[hat(x[3:]), x[:3].reshape((3, 1))],
                         [np.zeros((1, 4))]])
    else:
        raise ValueError('hat operation for vector with length not equal to 3 or 6 is not supported.')


def curly_hat(x):
    return np.block([[hat(x[3:]), hat(x[:3])],
                     [np.zeros((3, 3)), hat(x[3:])]])


def odot(x):
    if len(x) != 4:
        raise ValueError('odot operation for vector with length not equal to 4 is not supported.')
    return np.block([[x[3]*np.eye(3), -hat(x[0:3])],
                     [np.zeros((1, 6))]])


def pick_feature(feature):
    # pick observed landmarks given one feature map
    # Input:
    #       feature: 4 * M
    return np.where(feature[0, :] != -1)


def back_projection(observations, M, imu_T_cam, wld_T_imu):
    d = observations[0] - observations[2]
    fsub = -M[2, 3]
    z = fsub/d
    x = (observations[0] - M[0, 2]) * z / M[0, 0]
    y = (observations[1] - M[1, 2]) * z / M[1, 1]
    po = np.vstack((x, y, z, np.ones(x.shape)))       # homogeneous coord in camera frame
    pw = wld_T_imu @ imu_T_cam @ po
    return pw


def projection(cam_T_imu, imu_T_wld, mu_ldk):
    # projection function
    # Input:
    #        mu_ldk: dim (4, N_t)
    pc = cam_T_imu @ imu_T_wld @ mu_ldk
    return pc / pc[2]


def d_proj(q):
    # derivative of projection
    q = q.squeeze()
    return 1/q[2]*np.array([[1, 0, -q[0]/q[2], 0],
                            [0, 1, -q[1]/q[2], 0],
                            [0, 0, 0, 0],
                            [0, 0, -q[3]/q[2], 1]])

def check_static(z, z_hat):
    dist = np.linalg.norm(z - z_hat, axis=0)
    idx_dynamic = dist > 200        # if distance is larger than 200, then the feature is dynamic
    z_hat[:, idx_dynamic] = z[:, idx_dynamic]
    return z_hat

