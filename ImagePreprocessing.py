import torchvision.transforms as transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader

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
        # Calculate crop box boundaries
        left = max(nozzle_x - self.crop_size // 2, 0)
        upper = max(nozzle_y - self.crop_size // 2, 0)
        right = min(nozzle_x + self.crop_size // 2, image.width)
        lower = min(nozzle_y + self.crop_size // 2, image.height)

        cropped_img = image.crop((left, upper, right, lower))
        resized_img = cropped_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)

        return resized_img


class RandomCrop:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.size[1], image.size[0]  # the image will be after 320x320 reshaping so they are equal
        scale_factor = random.uniform(0.9, 1.0)
        crop_size = int(h*scale_factor)
        max_top = h - crop_size
        max_left = w - crop_size
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        cropped = image.crop((left, top, left + crop_size, top + crop_size))
        resized_cropped = cropped.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
        return resized_cropped


class HorizontalFlipColorJitter:   #apply with probability 0.5
    def __init__(self):
        self.brightness = 0.1
        self.contrast = 0.1
        self.saturation = 0.1
        self.hue = 0.1

    def __call__(self, image):
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        color_jitter = transforms.ColorJitter(
            brightness = self.brightness,
            contrast= self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        jittered_image = color_jitter(flipped)

        if random.random() < 0.5:
            return jittered_image
        else:
            return image


class TransformPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, image, **kwargs):
        for job in self.pipeline:
            if isinstance(job, CenteredCrop):
                image = job(image, kwargs['nozzle_x'], kwargs['nozzle_y'])
            else:
                image = job(image)
        return image


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
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        nozzle_x, nozzle_y = self.nozzle_coords[idx]

        if self.transform:
            if isinstance(self.transform, TransformPipeline):
                image = self.transform(image, nozzle_x=nozzle_x, nozzle_y=nozzle_y)
            else:
                image = self.transform(image)

        return image, label

