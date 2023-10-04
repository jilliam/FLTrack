import yaml
from synergy3DMM import SynergyNet

class FAwKF_SynergyNet_Interface():
    def __init__(self):
        self.model = SynergyNet()

    def get_landmarks_from_image(self, image, detected_faces):
        lmks = self.model.get_lmks(image, detected_faces)

        return lmks.T
