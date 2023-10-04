import cv2
import numpy as np
import face_alignment
from pykalman import KalmanFilter
import os
import glob
import ntpath
import shutil

class MyKF_gcv():
    def __init__(self, start, end, state_means, state_covariances):
        self.state_means = state_means
        self.state_covariances = state_covariances
        self.start = start
        self.end = end
        self.groupsize = end - start + 1

        O = np.zeros([3, 3])
        I = np.eye(3)

        block_list = []
        for i in range(self.groupsize+1):
            row = [I if (i==j or j==self.groupsize) else O for j in range(self.groupsize+1)]
            block_list.append(row)
        self.A = np.block(block_list)
        # print(self.A.shape)

        self.R = 0.1 * np.eye(3 * (self.groupsize+1))
        # print(self.R.shape)

        block_list = []
        for i in range(self.groupsize):
            row = [I if (i == j) else O for j in range(self.groupsize + 1)]
            block_list.append(row)
        self.C = np.block(block_list)
        # print(self.C.shape)

        self.Q = 0.1 * np.eye(3 * (self.groupsize))
        # print(self.Q.shape)

        self.KF = KalmanFilter(transition_matrices=self.A, transition_covariance=self.R,
                               observation_matrices=self.C, observation_covariance=self.Q)

    def set_observation_covariance(self, score_vector):
        a = np.repeat(score_vector, 3)
        self.Q = np.diag(a)

    def update(self, measurement):
        self.state_means, self.state_covariances = \
            self.KF.filter_update(self.state_means, self.state_covariances, measurement, observation_covariance=self.Q)

        return self.state_means

    def get_state_cov_norm(self):
        a = np.diag(self.state_covariances)[0::3]
        return a[0:-1]

class FAwKF_gcv():
    def __init__(self):
        self.kf_list = []
        self.len_track = 3

    def KFs_init(self, lmks_init):
        pass

    def KFs_update(self, lmks_pred, scores):
        lmks_corr = []
        for kf in self.kf_list:
            if scores is not None:
                kf.set_observation_covariance(1 - scores[kf.start:kf.end + 1])
            measurement = lmks_pred[kf.start:kf.end + 1]
            measurement = measurement.reshape(-1)
            corrected_state = list(kf.update(measurement))
            lmks_corr.append(corrected_state[0:-3])  # remove the velocity in the state, and then append
        lmks_corr = [item for sublist in lmks_corr for item in sublist]  # flatten list of list
        lmks_corr = np.array(lmks_corr).reshape(-1, 3)

        return lmks_corr

    def check_if_trigger_detector(self):
        """
            Method to compare the last state covariances of each KF. It counts the number of landmarks where the
            covariance has increased in the last M frames. If it is above a threshold, the face detector is restarted.
            :return: True or False, if the face detector needs to be restarted
        """
        norms = []
        for i in range(len(self.kf_list)):
            a = self.kf_list[i].get_state_cov_norm()
            norm = np.linalg.norm(a)
            norms.append(norm)

        norms = np.array(norms)
        # print("norms:", norms)

        self.last_cov[:, :(self.len_track - 1)] = self.last_cov[:, 1:self.len_track]
        self.last_cov[:, (self.len_track - 1)] = norms

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        norm_increase = np.zeros([len(self.kf_list), 1], dtype=bool)
        for i in range(len(self.kf_list)):
            norm_increase[i] = is_sorted(self.last_cov[i])

        num_cov_increase = np.count_nonzero(norm_increase)
        return num_cov_increase > int(len(self.kf_list) / 3)
        # return num_cov_increase > int(len(self.kf_list) / 5)

class FAwKF_gcv_1(FAwKF_gcv):
    def KFs_init(self, lmks_init):
        # Group 1 = (0-4)(5)
        init_state_means = np.append(lmks_init[0:5], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(0, 4, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 2 = (5-11)(7)
        init_state_means = np.append(lmks_init[5:12], [0, 0, 0])
        init_state_covariances = np.eye(24)
        kf = MyKF_gcv(5, 11, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 3 = (12-16)(5)
        init_state_means = np.append(lmks_init[12:17], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(12, 16, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 4 = (17-21)(5)
        init_state_means = np.append(lmks_init[17:22], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(17, 21, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 5 = (22-26)(5)
        init_state_means = np.append(lmks_init[22:27], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(22, 26, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 6 = (27-30)(4)
        init_state_means = np.append(lmks_init[27:31], [0, 0, 0])
        init_state_covariances = np.eye(15)
        kf = MyKF_gcv(27, 30, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 7 = (31-35)(5)
        init_state_means = np.append(lmks_init[31:36], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(31, 35, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 8 = (36-39)(4)
        init_state_means = np.append(lmks_init[36:40], [0, 0, 0])
        init_state_covariances = np.eye(15)
        kf = MyKF_gcv(36, 39, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 9 = (40-41)(2)
        init_state_means = np.append(lmks_init[40:42], [0, 0, 0])
        init_state_covariances = np.eye(9)
        kf = MyKF_gcv(40, 41, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 10 = (42-45)(4)
        init_state_means = np.append(lmks_init[42:46], [0, 0, 0])
        init_state_covariances = np.eye(15)
        kf = MyKF_gcv(42, 45, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 11 = (46-47)(2)
        init_state_means = np.append(lmks_init[46:48], [0, 0, 0])
        init_state_covariances = np.eye(9)
        kf = MyKF_gcv(46, 47, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 12 = (48-54)(7)
        init_state_means = np.append(lmks_init[48:55], [0, 0, 0])
        init_state_covariances = np.eye(24)
        kf = MyKF_gcv(48, 54, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 13 = (55-59)(5)
        init_state_means = np.append(lmks_init[55:60], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(55, 59, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 14 = (60-64)(5)
        init_state_means = np.append(lmks_init[60:65], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(60, 64, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 15 = (65-67)(3)
        init_state_means = np.append(lmks_init[65:68], [0, 0, 0])
        init_state_covariances = np.eye(12)
        kf = MyKF_gcv(65, 67, init_state_means, init_state_covariances)
        self.kf_list.append(kf)

        self.last_cov = np.zeros([len(self.kf_list), self.len_track])
        print("self.last_cov.shape:", self.last_cov.shape)

class FAwKF_gcv_2(FAwKF_gcv):
    def KFs_init(self, lmks_init):
        # Group 1 = (0-35)(36)
        init_state_means = np.append(lmks_init[0:36], [0, 0, 0])
        init_state_covariances = np.eye(111)
        kf = MyKF_gcv(0, 35, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 8 = (36-39)(4)
        init_state_means = np.append(lmks_init[36:40], [0, 0, 0])
        init_state_covariances = np.eye(15)
        kf = MyKF_gcv(36, 39, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 9 = (40-41)(2)
        init_state_means = np.append(lmks_init[40:42], [0, 0, 0])
        init_state_covariances = np.eye(9)
        kf = MyKF_gcv(40, 41, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 10 = (42-45)(4)
        init_state_means = np.append(lmks_init[42:46], [0, 0, 0])
        init_state_covariances = np.eye(15)
        kf = MyKF_gcv(42, 45, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 11 = (46-47)(2)
        init_state_means = np.append(lmks_init[46:48], [0, 0, 0])
        init_state_covariances = np.eye(9)
        kf = MyKF_gcv(46, 47, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 12 = (48-54)(7)
        init_state_means = np.append(lmks_init[48:55], [0, 0, 0])
        init_state_covariances = np.eye(24)
        kf = MyKF_gcv(48, 54, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 13 = (55-59)(5)
        init_state_means = np.append(lmks_init[55:60], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(55, 59, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 14 = (60-64)(5)
        init_state_means = np.append(lmks_init[60:65], [0, 0, 0])
        init_state_covariances = np.eye(18)
        kf = MyKF_gcv(60, 64, init_state_means, init_state_covariances)
        self.kf_list.append(kf)
        # Group 15 = (65-67)(3)
        init_state_means = np.append(lmks_init[65:68], [0, 0, 0])
        init_state_covariances = np.eye(12)
        kf = MyKF_gcv(65, 67, init_state_means, init_state_covariances)
        self.kf_list.append(kf)

        self.last_cov = np.zeros([len(self.kf_list), self.len_track])
        print("self.last_cov.shape:", self.last_cov.shape)
