import os

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision

from models import pvcNet
import utils

prefix_path = os.path.abspath(utils.__file__).rsplit('/',1)[0]
prefix_path = os.path.dirname(prefix_path)
print(prefix_path)


class JVCR(object):
    def __init__(self, stacks=4, blocks=1):
        super(JVCR, self).__init__()

        is_cuda = False
        # 68 points plus eye and mouth center
        nParts = 71
        # create model
        depth_res = [1, 2, 4, 64]
        resume_p2v2c = str(os.path.join(prefix_path, 'checkpoint/300wLP/model_p2v2c_300wLP.pth.tar'))
        print("==> creating model: stacks={}, blocks={}, z-res={}".format(stacks, blocks, depth_res))
        self.model = pvcNet(stacks, blocks, depth_res, nParts, resume_p2v2c=resume_p2v2c, is_cuda=is_cuda)
        self.model.resume_from_checkpoint()
        self.model.eval()

        # link for facial points
        skeletons = self.get_skeletons()

    def get_skeletons(self):
        skel = [[idx, idx + 1] for idx in range(16)] + \
               [[idx, idx + 1] for idx in range(17, 21)] + \
               [[idx, idx + 1] for idx in range(22, 26)] + \
               [[idx, idx + 1] for idx in range(36, 41)] + [[41, 36]] + \
               [[idx, idx + 1] for idx in range(42, 47)] + [[47, 42]] + \
               [[idx, idx + 1] for idx in range(27, 30)] + \
               [[idx, idx + 1] for idx in range(31, 35)] + \
               [[idx, idx + 1] for idx in range(48, 59)] + [[59, 48]] + \
               [[idx, idx + 1] for idx in range(60, 67)] + [[67, 60]]
        return skel

    def img_crop(self, image_tensor, center, scale):
        return utils.transforms.crop(image_tensor, center, scale, [256, 256])
        # return self.crop_fan(image_tensor, center, scale, 256.0)

    def transf_pred(self, pred_coord, center, scale):
        lm_pred = utils.transforms.transform_preds(pred_coord, center, scale, [256, 256], 256)

        lm_pred[:, 2] = -lm_pred[:, 2]

        z_mean = torch.mean(lm_pred[:, 2])
        lm_pred[:, 2] -= z_mean

        return lm_pred

    def get_lmks(self, image, bb):
        # enlarge the bbox a little and do a square crop
        # print(bb)

        bb[2:4] = bb[2:4] - bb[0:2]
        center = bb[0:2] + bb[2:4] / 2.
        scale = bb[2] / 200.
        scale *= 1.35        # 1.25
        # print(bb)

        image_tensor = torchvision.transforms.ToTensor()(image)
        # input image with size of 256x256
        input_tensor = self.img_crop(image_tensor, center, scale)

        pred_voxel, pred_coord = self.model.landmarkDetection(input_tensor.unsqueeze(0))
        pred_coord = pred_coord.data[0:68]

        lmks = self.transf_pred(pred_coord, center, scale)

        return lmks

    def crop_fan(self, image, center, scale, resolution=256.0):
        """Center crops an image or set of heatmaps
        Arguments:
            image {numpy.array} -- an rgb image
            center {numpy.array} -- the center of the object, usually the same as of the bounding box
            scale {float} -- scale of the face
        Keyword Arguments:
            resolution {float} -- the size of the output cropped image (default: {256.0})
        Returns:
            [type] -- [description]
        """  # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """
        image = utils.im_to_numpy(image)

        ul = self.transform_fan([1, 1], center, scale, resolution, True)
        br = self.transform_fan([resolution, resolution], center, scale, resolution, True)
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                               image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array(
            [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array(
            [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
        ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                            interpolation=cv2.INTER_LINEAR)

        newImg = utils.im_to_torch(newImg)

        return newImg

    def transform_fan(self, point, center, scale, resolution, invert=False):
        """Generate and affine transformation matrix.
        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.
        Arguments:
            point {torch.tensor} -- the input 2D point
            center {torch.tensor or numpy.array} -- the center around which to perform the transformations
            scale {float} -- the scale of the face/object
            resolution {float} -- the output resolution
        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]

        return new_point.int()


if __name__ == '__main__':
    pass
