# ECE 276A Project #3 - Visual-Inertial SLAM via EKF

Pengluo Wang,

University of California, San Diego, 2019

### Overview

Implement visual-inertial simultaneous localization and mapping (SLAM) using Extended Kalman filter. Synchronized measurements from a high-quality IMU and a stereo camera have been provided. The data is obtained from [KITTI dataset Raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential) and data pre-processing has been completed. The data includes:

* **IMU Measurements**: linear velocity ![](https://latex.codecogs.com/svg.latex?v_t&space;\in&space;\mathbb{R}^3) and angular velocity ![](https://latex.codecogs.com/svg.latex?\omega_t&space;\in&space;\mathbb{R}^3) measured in the body frame of the IMU 

* **Stereo Camera Images**: pixel coordinates  ![](https://latex.codecogs.com/svg.latex?z_t&space;\in&space;\mathbb{R}^{4\times&space;M}) of detected visual features with precomputed correspondences between the left and the right camera frames.

* **Time Stamps**: time stamps ![](https://latex.codecogs.com/svg.latex?\tau) in UNIX standard seconds-since-the-epoch.

* **Intrinsic Calibration**: stereo baseline ![](https://latex.codecogs.com/svg.latex?b) and camera calibration matrix ![](https://latex.codecogs.com/svg.latex?\mathbf{K}):

  <img src="https://latex.codecogs.com/svg.latex?\mathbf{K}=\begin{bmatrix}&space;fs_u&0&c_u\\0&fs_v&c_v\\0&0&1\end{bmatrix}" title="\mathbf{K}=\begin{bmatrix} fs_u&0&c_u\\0&fs_v&c_v\\0&0&1\end{bmatrix}" />

* **Extrinsic Calibration**: the transformation ![](https://latex.codecogs.com/svg.latex?_CT_I&space;\in&space;SE(3)) from the IMU to left camera frame.

### Requirements

- Python 3.7

### Installation

If you're using Conda for python environment management:

```
conda create -n vi_slam_env python==3.7
conda activate vi_slam_env
pip install -U pip
pip install -r requirements.txt
```

### Demo

Run

```
python main.py -d 0020
```

### Results

Dataset 0020:

<img src="results/0020_unf_c.png"/>

Dataset 0027:

<img src="results/0027_unf_c.png"/>

Dataset 0042:

<img src="results/0042_unf_c.png"/>