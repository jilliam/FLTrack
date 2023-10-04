import yaml
from TDDFA import TDDFA
import os

class FAwKF_3DDFA_V2_Interface():
    def __init__(self):
        # load config

        os.chdir('TDDFA_V2')

        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        self.tddfa = TDDFA(gpu_mode=False, **cfg)

        os.chdir('../')

    def get_landmarks_from_image(self, image, detected_faces):
        param_lst, roi_box_lst = self.tddfa(image, detected_faces)  # regress 3DMM params
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

        return ver_lst[0].T
