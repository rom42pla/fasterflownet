import os
from os.path import exists, join, isfile
from copy import deepcopy
import re

import cv2

import einops
import imageio
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from data.transform_utils import ConcatTransformSplitChainer, RandomGamma


def read_kitti_flow(flow_file):
    flow = cv2.imread(flow_file, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow


class KITTI2015FlowDataset(Dataset):
    def __init__(self, path, patch_size = (320,896),random_crop=False, center_crop=False, zoom=False, rotation=False, horizontal_flip=False,photometric_augmentations=False):
        assert exists(path)
        self.path = path
        self.photometric_augmentation = photometric_augmentations
        # data augmentations
        self.patch_size = patch_size
        assert not (random_crop and center_crop)
        assert isinstance(random_crop, bool)
        self.random_crop = random_crop
        assert isinstance(center_crop, bool)
        self.center_crop = center_crop
        assert isinstance(zoom, bool)
        self.zoom = zoom
        assert isinstance(rotation, bool)
        self.rotation = rotation
        assert isinstance(horizontal_flip, bool)
        self.horizontal_flip = horizontal_flip

        # assert isinstance(divide_optical_flow, bool)
        # self.divide_optical_flow = divide_optical_flow
        self.color_jitter = ConcatTransformSplitChainer([
            # uint8 -> PIL
            transforms.ToPILImage(),
            # PIL -> PIL : random hsv and contrast
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # PIL -> FloatTensor
            transforms.transforms.ToTensor(),
            RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
        ], from_numpy=False, to_numpy=False)
        self.samples_names = sorted(
            list({filename.split("_")[0] for filename in os.listdir(join(self.path, "image_2"))}))
        self.samples = []
        for sample_name in self.samples_names:
            sample = {
                "name": sample_name,
                "frame1": join(self.path, "image_2", f"{sample_name}_10.png"),
                "frame2": join(self.path, "image_2", f"{sample_name}_11.png")
            }
            flow_occ_path = join(self.path, "flow_occ", f"{sample_name}_10.png")
            if isfile(flow_occ_path):
                sample["flow_occ"] = flow_occ_path
            flow_noc_path = join(self.path, "flow_noc", f"{sample_name}_10.png")
            if isfile(flow_noc_path):
                sample["flow_noc"] = flow_noc_path
            self.samples += [sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = deepcopy(self.samples[i])
        for k, v in sample.items():
            if isfile(v) and v.endswith(".png") or v.endswith(".jpg") or v.endswith(".jpeg"):
                s = None
                if k.startswith("flow"):

                    s = torch.from_numpy(read_kitti_flow(v))
                else:
                    s = torch.from_numpy(cv2.imread(v, cv2.IMREAD_UNCHANGED).astype(float))
                    s = s / 255.0
                s = einops.rearrange(s, "h w c -> c h w")
                # s = torch.nn.functional.interpolate(s.unsqueeze(0), size=(384, 1280), mode='bilinear').squeeze()
                sample[k] = s.float()

        # data augmentations
        if self.random_crop:
            i, j, h, w = transforms.RandomCrop.get_params(
                sample["frame1"], output_size=self.patch_size)
            for k, v in sample.items():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                sample[k] = TF.crop(sample[k], i, j, h, w)

        if self.center_crop:
            for k, v in sample.items():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                sample[k] = transforms.CenterCrop(self.patch_size)(sample[k])

        if self.zoom:
            scale = 1 + (np.random.rand() * 2)
            i, j, h, w = transforms.RandomCrop.get_params(
                sample["frame1"],
                output_size=(int(self.patch_size[0] * (1 / scale)), int(self.patch_size[1] * (1 / scale))))
            for k, v in sample.items():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                sample[k] = transforms.Compose([
                    transforms.CenterCrop(
                        (int(self.patch_size[0] * (1 / scale)), int(self.patch_size[1] * (1 / scale)))),
                    transforms.Resize(self.patch_size)
                ])(sample[k])

        if self.horizontal_flip and np.random.rand() <= 0.5:
            for k, v in sample.items():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                sample[k] = TF.hflip(sample[k])

        if self.rotation:
            angle = np.random.rand() * 20.0 - 10.0
            for k, v in sample.items():
                if not isinstance(sample[k], torch.Tensor):
                    continue
                sample[k] = TF.rotate(sample[k], angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
        if self.photometric_augmentation:
            sample["frame1"], sample["frame2"] = self.color_jitter(sample["frame1"], sample["frame2"])
        if "flow_noc" in sample.keys():
            # if self.divide_optical_flow:
            #     sample["flow_noc"] = sample["flow_noc"]/20
            return sample["frame1"], sample["frame2"], sample["flow_noc"][[0, 1], :, :]
        else:
            return sample["frame1"], sample["frame2"]

    def get_random_sample(self):
        i = np.random.randint(0, len(self))
        return self[i]