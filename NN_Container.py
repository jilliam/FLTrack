import os
import numpy as np
import numpy.ma as ma
import glob
import os
import ntpath
from PIL import Image
from skimage import io
import cv2
import face_alignment
import sys
from os import path

class NN_Container():
    def __init__(self, dataset_path, npy_path, nn_choice):
        self.dataset_path = dataset_path
        # self.bboxes_detector = np.load(npy_path + '/bboxes.npy', allow_pickle=True).item()
        self.bboxes_detector = np.load(npy_path + '/bboxes_p2.npy', allow_pickle=True)  # JVCR
        print("keys in bboxes.npy:", self.bboxes_detector.keys())
        # self.sequences = sorted(glob.glob(dataset_path + "/*/"))
        seq_list = sorted(self.bboxes_detector.keys())
        # seq_list = [elem.encode("utf-8") for elem in seq_list]
        print("dataset_path:", dataset_path)
        print("seq_list:", seq_list)
        self.sequences = [os.path.join(dataset_path, elem) for elem in seq_list]
        self.nn_choice = nn_choice

        if self.nn_choice == "FAN":
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")

        elif self.nn_choice == "3DDFA_V2":
            sys.path.append(path.abspath('TDDFA_V2'))
            from FAwKF_3DDFA_V2_Interface import FAwKF_3DDFA_V2_Interface
            self.fa = FAwKF_3DDFA_V2_Interface()

        elif self.nn_choice == "SynergyNet":
            sys.path.append(path.abspath('SynergyNet'))
            from FAwKF_SynergyNet_Interface import FAwKF_SynergyNet_Interface
            self.fa = FAwKF_SynergyNet_Interface()

        elif self.nn_choice == "JVCR":
            sys.path.append(path.abspath('JVCR'))
            from FAwKF_JVCR_Interface import FAwKF_JVCR_Interface
            self.fa = FAwKF_JVCR_Interface()

        self.preds_3D = {}
        self.corrs_3D = {}

        self.prev_bbox = [0, 0, 0, 0]

    def get_bbox_from_detector(self, seq, frame):
        # seq = ntpath.basename(seq[0:-1])
        seq = os.path.basename(seq)
        return self.bboxes_detector[seq][frame]

    def get_sequences(self):
        return self.sequences

    def get_seq_len(self, seq):
        ext = ['png', 'jpg', 'jpeg']
        files = []
        [files.extend(glob.glob(os.path.join(seq, '*.' + e))) for e in ext]
        return len(files)

    def predict(self, img_path, bb):
        if self.nn_choice == "FAN":
            image = io.imread(img_path)
            bb = np.append(bb, 1)
            FAN_out = self.fa.get_landmarks_from_image(image, return_landmark_score=True, detected_faces=[bb])
            lmks_pred = FAN_out[0][0]
            scores = FAN_out[1][0]
            return lmks_pred, scores

        elif self.nn_choice == "3DDFA_V2" or self.nn_choice == "SynergyNet":
            image = cv2.imread(img_path)
            lmks_pred = self.fa.get_landmarks_from_image(image, detected_faces=[bb])
            return lmks_pred, None

        elif self.nn_choice == "JVCR":
            image = Image.open(img_path).convert('RGB')
            lmks_pred = self.fa.get_landmarks_from_image(image, detected_faces=bb)
            return lmks_pred, None



