from tqdm import tqdm
import json
from NPY_Loader import *
from NN_Container import *
from FAwKF_ca import *
from FAwKF_cv import *
from FAwKF_gcv import *
import torch


def Run_offline(npy_path, nn_choice):

    myLorder = NPY_Loader(npy_path, nn_choice + ".npy")

    sequences = myLorder.get_sequences()
    print("sequences:", sequences)
    dict_out = {}

    for seq in tqdm(sequences):
        corr_arr_list = []
        fawkf = FAwKF_gcv_2()
        for frame in range(myLorder.get_seq_len()):
            lmks_pred = myLorder.get_lmks_pred(seq, frame)
            if frame == 0:
                fawkf.KFs_init(lmks_pred)
                lmks_corr = lmks_pred
            else:
                lmks_corr = fawkf.KFs_update(lmks_pred, scores=None)

            corr_arr_list.append(lmks_corr)
        corr_arr_out = np.stack(corr_arr_list, axis=2)
        seqId = os.path.basename(seq)
        dict_out[seqId] = corr_arr_out
    np.save('offline_' + nn_choice + '.npy', dict_out)


def Run_online(dataset_path, npy_path, nn_choice):
    myNN = NN_Container(dataset_path, npy_path, nn_choice)

    keyframe_counter_list = {}
    for seq in myNN.get_sequences():
        print("seq:", seq)
        pred_arr_list = []
        corr_arr_list = []

        fawkf = FAwKF_gcv_2()

        is_keyframe = True
        keyframe_counter = 0
        print('myNN.get_seq_len(seq)', myNN.get_seq_len(seq))
        for frame in range(myNN.get_seq_len(seq)):
            print("frame:", frame)
            # filename = f"{frame + 1:06d}.jpg"
            filename = '{0:06d}'.format(frame+1) + '.jpg'  # Menpo
            # filename = '{0:04d}'.format(frame) + '.png'    # ParFace
            img_path = os.path.join(seq, filename)

            bb = myNN.get_bbox_from_detector(seq, frame)

            if is_keyframe:
                # Keyframes, use detector
                print("Keyframes, use detector")
                keyframe_counter = keyframe_counter + 1
                if not np.any(bb):
                    # No face detected in the frame, set pred as masked array
                    print("XXX: bb:{}, prebb:{}".format(bb, myNN.prev_bbox))
                    x = np.zeros((68, 3))
                    m = np.ones((68, 3), dtype=bool)
                    lmks_pred = ma.masked_array(x, mask=m)
                    scores = None
                    pred_arr_list.append(lmks_pred)
                    myNN.prev_bbox = [0, 0, 0, 0]
                else:
                    # ================================================================================================
                    new_y1 = bb[1] * 0.8 + bb[3] * 0.2
                    bb[1] = new_y1
                    lmks_pred, scores = myNN.predict(img_path, bb)
                    pred_arr_list.append(lmks_pred)
                    # ================================================================================================
            else:
                # Non-Keyframes, use the bbox from previous frame
                print("Non-Keyframes, use the bbox from previous frame")
                if not np.any(myNN.prev_bbox):
                    if not np.any(bb):
                        # No face detected in the previous and current frame, set pred as masked array
                        print("XXX: bb:{}, prebb:{}".format(bb, myNN.prev_bbox))
                        x = np.zeros((68, 3))
                        m = np.ones((68, 3), dtype=bool)
                        lmks_pred = ma.masked_array(x, mask=m)
                        scores = None
                        pred_arr_list.append(lmks_pred)
                        myNN.prev_bbox = [0, 0, 0, 0]
                    else:
                        # No face detected in the previous frame BUT current frame, trigger detector
                        # ================================================================================================
                        keyframe_counter = keyframe_counter + 1

                        new_y1 = bb[1] * 0.8 + bb[3] * 0.2
                        bb[1] = new_y1
                        lmks_pred, scores = myNN.predict(img_path, bb)
                        pred_arr_list.append(lmks_pred)
                        # ================================================================================================
                else:
                    # ================================================================================================
                    lmks_pred, scores = myNN.predict(img_path, myNN.prev_bbox)
                    pred_arr_list.append(lmks_pred)
                    # ================================================================================================

            if frame == 0:
                fawkf.KFs_init(lmks_pred)
                lmks_corr = lmks_pred if not ma.is_masked(lmks_pred) else lmks_pred.filled(0)
            else:
                lmks_corr = fawkf.KFs_update(lmks_pred, scores)

            corr_arr_list.append(lmks_corr)
            if torch.is_tensor(lmks_corr):
                lmks_corr = lmks_corr.numpy()
            # print("lmks_corr", lmks_corr)
            myNN.prev_bbox = np.array([np.min(lmks_corr[:, 0]), np.min(lmks_corr[:, 1]),
                                       np.max(lmks_corr[:, 0]), np.max(lmks_corr[:, 1])])

            # new_y1 = myNN.prev_bbox[3] - (myNN.prev_bbox[3] - myNN.prev_bbox[1]) * 1.25
            # myNN.prev_bbox[1] = new_y1 if new_y1 >= 0 else 0

            if nn_choice == "FAN" and frame > 0:  # Restart detector if covariance increases consecutively after a number of frames
                is_keyframe = fawkf.check_if_trigger_detector()
                print("AM, Restart detector", is_keyframe)
            else:
                # is_keyframe = True if (frame + 1) % 5 == 0 else False
                is_keyframe = True if (frame + 1) % 10 == 0 else False
                # is_keyframe = True if (frame + 1) % 30 == 0 else False

            if np.any(myNN.prev_bbox):
                if (myNN.prev_bbox[3] - myNN.prev_bbox[1]) < 20 or (myNN.prev_bbox[2] - myNN.prev_bbox[0]) < 20:
                    is_keyframe = True
                    myNN.prev_bbox = [0, 0, 0, 0]

        pred_arr_out = np.stack(pred_arr_list, axis=2)
        corr_arr_out = np.stack(corr_arr_list, axis=2)

        seqId = os.path.basename(seq)
        myNN.preds_3D[seqId] = pred_arr_out
        myNN.corrs_3D[seqId] = corr_arr_out
        keyframe_counter_list[seqId] = keyframe_counter

    np.save('online_' + nn_choice + '.npy', myNN.corrs_3D, allow_pickle=True)
    with open('keyframe_counter_' + nn_choice + '_jvcr.json', "w") as jsonfile:
        json.dump(keyframe_counter_list, jsonfile)


if __name__ == '__main__':
    dataset_path = "test"
    npy_path = "npys"
    nn_choice = 'JVCR'  # "FAN" - "SynergyNet" - "3DDFA_V2"
    online = True

    if online:
        Run_online(dataset_path, npy_path, nn_choice)
    else:
        Run_offline(npy_path, nn_choice)




