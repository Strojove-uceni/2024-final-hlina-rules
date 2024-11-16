import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from ImagePreprocessing import CustomDataset, TransformPipeline
from torch.utils.data import DataLoader
from ImagePreprocessing import RandomRotate, RandomPerspectiveTransform, CenteredCrop, RandomCrop, HorizontalFlipColorJitter

DATASET_DIR = 'CambridgeData'
IMAGES_DIR = os.path.join(DATASET_DIR, "print0")
LABELS_PATH = os.path.join(DATASET_DIR, "caxton_dataset_full.csv")
df = pd.read_csv(LABELS_PATH)


image_paths = df['img_path'].values
image_paths = [os.path.join(DATASET_DIR, path) for path in df['img_path'].values] #create absolute path to avoid problems

labels = df[['flow_rate', 'feed_rate', 'z_offset', 'target_hotend']].values
nozzle_coords = df[['nozzle_tip_x', 'nozzle_tip_y']].values

image_train, image_temp, labels_train, labels_temp, nozzle_coords_train, nozzle_coords_temp = train_test_split(
    image_paths, labels, nozzle_coords, test_size=0.3, random_state=42
)

image_val, image_test, labels_val, labels_test, nozzle_coords_val, nozzle_coords_test = train_test_split(
    image_temp, labels_temp, nozzle_coords_temp, test_size=0.5, random_state=42
)

train_tf_pipeline = TransformPipeline([
    RandomRotate(), RandomPerspectiveTransform(), CenteredCrop(), RandomCrop(), HorizontalFlipColorJitter()
])


train_dataset = CustomDataset(image_train, labels_train, nozzle_coords_train, transform=train_tf_pipeline)
val_dataset = CustomDataset(image_val, labels_val, nozzle_coords_val, transform=None)
test_dataset = CustomDataset(image_test, labels_test, nozzle_coords_test, transform=None)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

