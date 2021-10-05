import argparse
import os

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage.morphology import binary_fill_holes
from torchvision import transforms as T
from segmentation.loader import correct_dims

from detection.YOLODetector import YOLODetector


def prepare_input(*images, transform, device):
    output_images = correct_dims(images)
    output_images = [transform(i) for i in output_images]
    output_images = [i.unsqueeze(0) for i in output_images]
    output_images = [i.to(torch.device(device)) for i in output_images]
    return output_images


def preprocess_nii(img, mask=None):
    if (img.sum() == 0): return img
    if (mask is None):
        cp = img
        mask = np.where(img > 0)
    else:
        mask = binary_fill_holes(mask).astype(int)
        cp = mask
        mask = np.where(mask > 0)
    try:
        img[np.isinf(img)] = 0
        img[np.isnan(img)] = 0
        minp = np.percentile(img[mask], 5)
        maxp = np.percentile(img[mask], 95)
        img[(cp > 0) & (img < minp)] = minp
        img[(cp > 0) & (img > maxp)] = maxp
        res = np.zeros(img.shape)
        res[mask] = (img[mask] - minp) / (maxp - minp)
    except:
        return np.zeros(img.shape)
    return res


def is_file(dirarg):
    """ Type for argparse - checks that directory exists.
    """
    if not os.path.isfile(dirarg):
        raise argparse.ArgumentError(
            "The file '{0}' does not exist!".format(dirarg))
    return dirarg


def get_args():
    parser = argparse.ArgumentParser("This program produced masks from BBUNet")

    # Required arguments
    parser.add_argument(
        "-w", "--model_weight", required=True, type=is_file,
        help="model model_weight .pt")
    parser.add_argument(
        "-i", "--input_image", required=True, type=is_file,
        help="location of source nii file")
    parser.add_argument(
        "-m", "--model", required=True, type=str, choices=['unet', 'bbunet'], default='unet',
        help="location of source nii file")
    parser.add_argument("--device", type=str, default='cpu', help="location of source nii file")
    parser.add_argument('--use_bbox', dest='usebbox', action='store_true',
                        help="use provided bounding box if true or predict one if false")
    parser.add_argument('--bbox', type=is_file, help="The bounding-boxes to use if provided. Nii file")
    parser.add_argument('--yolo_weight', type=is_file, help="detection pt file if bounding_boxes are predicted")
    parser.add_argument('--postprocess', dest='postprocess', action='store_true',
                        help="post_process yolo masks")
    parser.add_argument('--save_yolo', dest='saveyolo', action='store_true',
                        help="save yolo masks")
    parser.add_argument('--use_masking', action='store_true',help="use detection in masking")
    parser.add_argument(
        "-br", "--brainmask", required=False, type=is_file,
        help="location of brain mask nii file")
    parser.add_argument(
        "-o", "--outdir", required=True,
        help="the output directory")
    args = parser.parse_args()
    kwargs = vars(args)
    if (kwargs['use_bbox'] and not kwargs['bbox']):
        raise ("You must provide the bounding boxes if you want to use_bbox")
    elif (not (kwargs['use_bbox'] or kwargs['yolo_weights'])):
        raise ("You must provide the yolo_weights if you dont have the bounding_boxes")
    return kwargs


inputs = get_args()
transform = T.ToTensor()

model = torch.load(inputs['model_weight'])
model = model.to(device=torch.device(inputs['device']))

brainmask = nib.load(inputs['brainmask']).get_fdata() if inputs.get('brainmask', False) else None
img_data = preprocess_nii(nib.load(inputs['input_image']).get_fdata(), mask=brainmask)
if (inputs['use_bbox']):
    bbox = nib.load(inputs['bbox']).get_fdata()
else:
    detector = YOLODetector(weights=inputs['yolo_weights'], post_process=inputs['postprocess'])
    niigz, bbox = detector.detect(img_data)
    if (inputs['save_yolo']):
        niigz.to_filename(os.path.join(inputs['outdir'], "yolo_" + os.path.basename(inputs['input_image'])))

h, w, d = img_data.shape
reconstruct = np.zeros(img_data.shape)
for z in range(d):
    img = np.zeros((h, w, 3))
    for i in range(3):
        img[:, :, i] = img_data[:, :, z]

    img = (img * 255).astype(np.uint8)

    if (inputs['model'] == 'bbunet'):
        mask = bbox[:, :, z]
        if (mask.sum() == 0):
            continue
        img, bbox_mask = prepare_input([img, mask], transform, inputs['device'])
        y_out = model(img, bbox=bbox_mask).cpu().data.numpy()
    else:
        img = prepare_input([img], transform, inputs['device'])[0]
        y_out = model(img, bbox=None).cpu().data.numpy()
    reconstruct[:, :, z] = y_out[0, 1, :, :]
if (inputs['use_masking']):
    reconstruct[bbox == 0] = 0

nib.Nifti1Image(reconstruct, nib.load(inputs['input_image']).affine).to_filename(
    os.path.join(inputs['outdir'], "segmentation_" + os.path.basename(inputs['input_image'])))
