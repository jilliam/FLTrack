import numpy as np
import numpy.ma as ma
import glob
import os

class NPY_Loader():
    def __init__(self, npy_path, npy_name):
        # self.pred_land = np.load(os.path.join(npy_path, npy_name), allow_pickle=True).item()
        self.pred_land = np.load(os.path.join(npy_path, npy_name), allow_pickle=True)     # JVCR
        self.sequences = list(sorted(self.pred_land.keys()))
        self.len_seq = next(iter(self.pred_land.values())).shape[2]

    def get_sequences(self):
        return self.sequences

    def get_seq_len(self):
        return self.len_seq

    def get_lmks_pred(self, seq, frame):

        if np.any(self.pred_land[seq][:, :, frame]):
            lmks_pred = self.pred_land[seq][:, :, frame]
        else:
            print("Warning! No face detected in frame {} of seq {}".format(frame, seq))
            x = np.zeros((68, 3))
            m = np.ones((68, 3), dtype=bool)
            lmks_pred = ma.masked_array(x, mask=m)

        return lmks_pred

