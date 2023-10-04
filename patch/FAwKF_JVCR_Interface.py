from JVCR_FA import JVCR
import os


class FAwKF_JVCR_Interface():
    def __init__(self):
        self.model = JVCR()

    def get_landmarks_from_image(self, image, detected_faces):
        lmks = self.model.get_lmks(image, detected_faces)

        # return lmks.T
        return lmks
