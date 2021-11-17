import os
from os.path import exists, join, isfile
from copy import deepcopy
import re

import einops
import imageio
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from data.transform_utils import RandomGamma, ConcatTransformSplitChainer


class SINTELDataset(Dataset):
    def __init__(self, path, split, db_type="final", patch_size = (320,896),random_crop=False, center_crop=False, zoom=False, rotation=False,
                 horizontal_flip=False,photometric_augmentations=False):
        assert exists(path)
        self.path = path

        assert split in {"train", "test"}
        self.split = split
        self.split_path = join(self.path, "training" if self.split == "train" else "test")

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
        self.photometric_augmentation = photometric_augmentations
        self.samples = []
        if db_type == "both":
            db_type="clean"
            for scene in os.listdir(join(self.split_path, db_type)):
                for frame1_filename in os.listdir(join(self.split_path, db_type, scene)):
                    frame1_number = re.search("[0-9]+", frame1_filename)[0]
                    frame2_filename = frame1_filename.replace(frame1_number, str(int(frame1_number) + 1).rjust(4, "0"))
                    if not exists(join(self.split_path, db_type, scene, frame2_filename)):
                        continue
                    sample = {
                        "name": join(scene, frame1_filename),
                        "frame1": join(self.split_path, db_type, scene, frame1_filename),
                        "frame2": join(self.split_path, db_type, scene, frame2_filename)
                        # "flow": join(self.path, "FlyingChairs_release", "data", f"{sample_name}_flow.flo"),
                    }
                    if exists(join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))):
                        sample["flow"] = join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))
                    self.samples += [sample]
            db_type = "final"
            for scene in os.listdir(join(self.split_path, db_type)):
                for frame1_filename in os.listdir(join(self.split_path, db_type, scene)):
                    frame1_number = re.search("[0-9]+", frame1_filename)[0]
                    frame2_filename = frame1_filename.replace(frame1_number, str(int(frame1_number) + 1).rjust(4, "0"))
                    if not exists(join(self.split_path, db_type, scene, frame2_filename)):
                        continue
                    sample = {
                        "name": join(scene, frame1_filename),
                        "frame1": join(self.split_path, db_type, scene, frame1_filename),
                        "frame2": join(self.split_path, db_type, scene, frame2_filename)
                        # "flow": join(self.path, "FlyingChairs_release", "data", f"{sample_name}_flow.flo"),
                    }
                    if exists(join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))):
                        sample["flow"] = join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))
                    self.samples += [sample]
        else:
            for scene in os.listdir(join(self.split_path, db_type)):
                for frame1_filename in os.listdir(join(self.split_path, db_type, scene)):
                    frame1_number = re.search("[0-9]+", frame1_filename)[0]
                    frame2_filename = frame1_filename.replace(frame1_number, str(int(frame1_number) + 1).rjust(4, "0"))
                    if not exists(join(self.split_path, db_type, scene, frame2_filename)):
                        continue
                    sample = {
                        "name": join(scene, frame1_filename),
                        "frame1": join(self.split_path, db_type, scene, frame1_filename),
                        "frame2": join(self.split_path, db_type, scene, frame2_filename)
                        # "flow": join(self.path, "FlyingChairs_release", "data", f"{sample_name}_flow.flo"),
                    }
                    if exists(join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))):
                        sample["flow"] = join(self.split_path, "flow", scene, frame1_filename.replace(".png", ".flo"))
                    self.samples += [sample]
        self.center_crop=transforms.CenterCrop(self.patch_size)
        self.random_gamma = RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True)
        self.color_jitter = ConcatTransformSplitChainer([
                # uint8 -> PIL
                transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                transforms.transforms.ToTensor(),
                RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=False, to_numpy=False)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = deepcopy(self.samples[i])

        for k, v in sample.items():
            if isfile(v) and v.endswith(".png") or v.endswith(".flo"):
                sample[k] = self.read_file(v)

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
                sample[k] = self.center_crop(sample[k])

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
           sample["frame1"],sample["frame2"]=self.color_jitter(sample["frame1"],sample["frame2"])


        if "flow" in sample.keys():
            return sample["frame1"], sample["frame2"], sample["flow"]
        else:
            return sample["frame1"], sample["frame2"]

    def read_file(self, path):
        if path.endswith('.flo'):
            return self.read_flo(path)
        elif path.endswith('.png'):
            return self.read_png(path)
        else:
            raise Exception(f"Unknown extension for file {path}")

    def read_flo(self, path):
        f = open(path, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
        flow = torch.from_numpy(flow.astype(np.float32))
        flow = einops.rearrange(flow, "h w c -> c h w")
        return flow

    def read_png(self, path):
        image = imageio.imread(path)
        image = torch.from_numpy(image)
        image = image.float() / 255
        image = einops.rearrange(image, "h w c -> c h w")
        return image

    def get_random_sample(self):
        i = np.random.randint(0, len(self))
        return self[i]
