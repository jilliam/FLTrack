import cv2
import numpy as np
import face_alignment
from pykalman import KalmanFilter
import os
import glob
import ntpath
import shutil

class MyKF_ca():
    def __init__(self, state_means, state_covariances):
        self.state_means = state_means
        self.state_covariances = state_covariances

        O = np.zeros([3, 3])
        I = np.eye(3)
        self.A = np.block([[I, I, 0.5*I],
                            [O, I, I],
                            [O, O, I]])
        self.R = 0.1 * np.eye(9)
        self.C = np.block([I, O, O])
        self.Q = 0.1 * np.eye(3)

        self.KF = KalmanFilter(transition_matrices = self.A, transition_covariance = self.R,
                               observation_matrices = self.C, observation_covariance = self.Q)

    def set_observation_covariance(self, score_vector):
        a = np.repeat(score_vector, 3)
        self.Q = np.diag(a)

    def update(self, measurement):
        self.state_means, self.state_covariances = \
            self.KF.filter_update(self.state_means, self.state_covariances, measurement, observation_covariance=self.Q)

        return self.state_means

    def get_state_cov_norm(self):
        return np.diag(self.state_covariances)[0]

class FAwKF_ca():
    def __init__(self):
        self.kf_list = []

    def KFs_init(self, lmks_init):
        for i in range(68):
            init_state_means = np.append(lmks_init[i], [0, 0, 0, 0, 0, 0])
            init_state_covariances = np.eye(9)
            kf = MyKF_ca(init_state_means, init_state_covariances)
            self.kf_list.append(kf)

        self.len_track = 3
        self.last_cov = np.zeros([68, self.len_track])
        print("self.last_cov.shape:", self.last_cov.shape)

    def KFs_update(self, lmks_pred, scores):
        lmks_corr = []
        for i in range(68):
            if scores is not None:
                self.kf_list[i].set_observation_covariance(1 - scores[i])
            measurement = lmks_pred[i]
            corrected_state = list(self.kf_list[i].update(measurement))
            lmks_corr.append(corrected_state[0:3])
        lmks_corr = np.array(lmks_corr)

        return lmks_corr

    def check_if_trigger_detector(self):
        """
            Method to compare the last state covariances of each KF. It counts the number of landmarks where the
            covariance has increased in the last M frames. If it is above a threshold, the face detector is restarted.
            :return: True or False, if the face detector needs to be restarted
        """
        norms = np.zeros([len(self.kf_list)])
        for i in range(len(self.kf_list)):
            norms[i] = self.kf_list[i].get_state_cov_norm()

        self.last_cov[:, :(self.len_track - 1)] = self.last_cov[:, 1:self.len_track]
        self.last_cov[:, (self.len_track - 1)] = norms

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        norm_increase = np.zeros([len(self.kf_list), 1], dtype=bool)
        for i in range(len(self.kf_list)):
            norm_increase[i] = is_sorted(self.last_cov[i])

        num_cov_increase = np.count_nonzero(norm_increase)
        return num_cov_increase > int(len(self.kf_list) / 5)



