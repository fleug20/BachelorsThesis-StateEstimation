import numpy as np
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous


def get_circular_trajectory():
    return ThreeDCircularTraj(center=np.array([0, 0, 1.5]),
                              radius=np.array([1.5, 1.5, 0]),
                              freq=np.array([0.2, 0.2, 0.2]),
                              )


def get_varying_height_circular_trajectory():
    return ThreeDCircularTraj(center=np.array([0, 0, 1.5]),
                              radius=np.array([1.5, 1.5, 0.5]),
                              freq=np.array([0.2, 0.2, 0.2]),
                              )


def get_fast_circular_trajectory():
    return ThreeDCircularTraj(center=np.array([0, 0, 0.8]),
                              radius=np.array([1.5, 1.5, 0]),
                              freq=np.array([0.4, 0.4, 0.4]))


def get_figure8_trajectory():
    return TwoDLissajous(A=2, B=2, a=0.5, b=1.0, delta=np.pi / 2, height=1)


def get_vertical_oscillation_trajectory():
    return ThreeDCircularTraj(center=np.array([0, 0, 1.5]),
                              radius=np.array([0, 0, 1]),
                              freq=np.array([0, 0, 0.3]))
