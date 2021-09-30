from time import time

import nibabel as nib
import numpy as np
import torch
from models.experimental import attempt_load
from scipy.ndimage.morphology import binary_closing, binary_opening
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class TumorDetector(object):

    def __init__(self, weights, img_size=512, conf_thres=0.3, post_process=False, kernel_size=7):
        self.device = select_device()
        self.model = attempt_load(weights, map_location=self.device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.post_process = post_process
        self.kernel_size = kernel_size

    def detect(self, input_nib):
        input_mat = input_nib.get_fdata()
        t0 = time()
        mask = np.zeros(input_nib.shape)

        imgs = self.prepare_image(input_mat)
        for idx, img in enumerate(imgs):
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, self.conf_thres, 0.5,
                                       multi_label=False, classes=[], agnostic=False)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (input_mat.shape[0], input_mat.shape[1])).round()
                    # Write results
                    for *xyxy, conf, cls in det:
                        xmin, ymin, xmax, ymax = [int(e.item()) for e in xyxy]
                        mask[ymin:ymax, xmin:xmax, idx] = conf.cpu()
        if self.post_process:
            mask = self.post_processing(mask, self.kernel_size)
        return nib.Nifti1Image(mask, input_nib.affine), mask

    def prepare_image(self, input_3d):

        imgs = []
        for z in range(input_3d.shape[2]):
            img = np.zeros((input_3d.shape[0], input_3d.shape[1], 3))
            for c in range(3):
                img[:, :, c] = input_3d[:, :, z]
            img = letterbox(img, new_shape=self.img_size)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            imgs.append(img)

        return imgs

    def post_processing(self, mask, kernel_size):
        mask = binary_closing(mask, structure=np.ones((1, 1, kernel_size)))
        mask = binary_opening(mask, structure=np.ones((1, 1, kernel_size)))
        return mask
