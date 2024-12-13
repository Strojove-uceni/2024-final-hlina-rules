import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
import torch
from os.path import exists

class RandomRotate:
    def __init__(self):
        pass

    def __call__(self, image):
        transform = transforms.RandomRotation(degrees=(-10,10))
        return transform(image)

class RandomPerspectiveTransform:    #applied with the 0.1 probability
    def __init__(self):
        pass

    def __call__(self, image):
        transform = transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
        if random.random() < 0.1:
            return transform(image)
        else:
            return image

class CenteredCrop:
    def __init__(self, crop_size=100, output_size=320):
        self.crop_size = crop_size
        self.output_size = output_size

    def __call__(self, image, nozzle_x, nozzle_y):
        # Ensure nozzle_x and nozzle_y are scalars (already ensured in CustomDataset)
        left = int(nozzle_x - self.crop_size / 2)
        upper = int(nozzle_y - self.crop_size / 2)
        right = int(nozzle_x + self.crop_size / 2)
        lower = int(nozzle_y + self.crop_size / 2)

        cropped_img = image.crop((left, upper, right, lower))
        resized_img = cropped_img.resize((self.output_size, self.output_size), Image.LANCZOS)

        return resized_img

class RandomCrop:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.size[1], image.size[0]  # the image will be after 320x320 reshaping so they are equal
        scale_factor = random.uniform(0.9, 1.0)
        crop_size = int(h * scale_factor)
        max_top = h - crop_size
        max_left = w - crop_size
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        cropped = image.crop((left, top, left + crop_size, top + crop_size))
        resized_cropped = cropped.resize((self.output_size, self.output_size), Image.LANCZOS)
        return resized_cropped

class HorizontalFlipColorJitter:  # apply with probability 0.5
    def __init__(self):
        self.brightness = 0.1
        self.contrast = 0.1
        self.saturation = 0.1
        self.hue = 0.1

    def __call__(self, image):
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        color_jitter = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        jittered_image = color_jitter(flipped)

        if random.random() < 0.5:
            return jittered_image
        else:
            return image

class TransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms  # List of transformations

    def __call__(self, image, nozzle_x=None, nozzle_y=None):
        print("Starting image transformations")
        for transform in self.transforms:
            if isinstance(transform, CenteredCrop):
                if nozzle_x is not None and nozzle_y is not None:
                    image = transform(image, nozzle_x, nozzle_y)
                else:
                    raise ValueError("nozzle_x and nozzle_y must be provided for CenteredCrop")
            else:
                image = transform(image)
        print("Image transformations finished")
        return image


class IterateDataset(Dataset):  #simple Dataset that iterates through images, use for compuatation of mean and std channels
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not exists(image_path):
            print(f"Image not found: {image_path}.")
            return self.__getitem__((idx + 1) % len(self))
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


class CalculateMeanStd:
    def __init__(self, image_paths, batch_size=32):
        self.image_paths = image_paths
        self.dataset = IterateDataset(self.image_paths)
        self.data_loader = DataLoader(self.dataset, batch_size= batch_size, shuffle = False)
        self.mean = torch.zeros(3)
        self.std = torch.zeros(3)
        self.num_samples = 0

    def __call__(self):
        self.mean.zero_()
        self.std.zero_()
        self.num_samples = 0
        for images in self.data_loader:
            num_batch_samples = images.size(0)
            images = images.view(num_batch_samples, 3, -1)
            self.mean += images.mean(2).sum(0)
            self.std += images.std(2).sum(0)
            self.num_samples += num_batch_samples

        mean = self.mean/self.num_samples
        std = self.std/self.num_samples

        return mean, std

class NormalizeChannels:
    def __init__(self, image_paths):
        mean_std_calculator=CalculateMeanStd(image_paths)
        self.mean, self.std = mean_std_calculator()
        self.to_tensor = transforms.ToTensor()
    def __call__(self, image):
        image = self.to_tensor(image)
        normalize = transforms.Normalize(self.mean, self.std)
        normalized = normalize(image)
        return normalized


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, nozzle_coords, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.nozzle_coords = nozzle_coords
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not exists(image_path):
            print(f"path does not exist {image_path}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        nozzle_x, nozzle_y = self.nozzle_coords[idx]

        if self.transform:
            if isinstance(self.transform, TransformPipeline):
                image = self.transform(image, nozzle_x=nozzle_x, nozzle_y=nozzle_y)
            else:
                image = self.transform(image)

        return image, label





