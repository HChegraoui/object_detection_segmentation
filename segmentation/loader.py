import os
from typing import Callable

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms \
    as T


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False,
                 use_bbox: bool = True) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.use_bbox = use_bbox
        if (self.use_bbox):
            self.bbox_path = os.path.join(dataset_path, "bbox")
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            #self.joint_transform = lambda x, y, z: (to_tensor(x), to_tensor(y), to_tensor(z))
            self.joint_transform = lambda *xs: tuple([to_tensor(x) for x in xs])

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        bbox = None
        image_filename = self.images_list[idx]
        # read image
        image = io.imread(os.path.join(self.input_path, image_filename))
        # read mask image
        mask_ = io.imread(os.path.join(self.output_path, image_filename))
        mask = np.zeros((mask_.shape[0], mask_.shape[1]))
        mask[:, :] = mask_ != 0
        mask = mask.astype(np.uint8)

        if (self.use_bbox):
            bbox = io.imread(os.path.join(self.bbox_path, image_filename))
            bbox = bbox.astype(np.uint8)
        # correct dimensions if needed
        if (self.use_bbox):
            image, mask, bbox = correct_dims(image, mask, bbox)
        else:
            image, mask = correct_dims(image, mask)

        if self.joint_transform:
            if (self.use_bbox):
                image, mask, bbox = self.joint_transform(image, mask, bbox)
            else:
                image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        if(self.use_bbox):
            return image, mask, bbox, image_filename
        return image, mask, image_filename
