import os
import pandas as pd
from sklearn.model_selection import train_test_split
from DataPreprocessing import CustomDataset, TransformPipeline
from torch.utils.data import DataLoader
from DataPreprocessing import RandomRotate, RandomPerspectiveTransform, CenteredCrop, RandomCrop, \
    HorizontalFlipColorJitter
from DataPreprocessing import CalculateMeanStd, NormalizeChannels
import torchvision.transforms as transforms
from RosNetModel import RosNet
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import time

DATASET_DIR = '/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/caxton_dataset'

print_dirs = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) if d.startswith('print')]

all_image_paths = []
all_labels = []
all_nozzle_coords = []

print("preparing directories")
for print_dir in print_dirs:
    csv_path = os.path.join(print_dir, 'print_log_filtered_classification3.csv')
    print(csv_path)
    if not os.path.exists(csv_path):
        print(f"Skipping {print_dir}: Missing 'print_log_filtered_classification3.csv'")
        continue

    df = pd.read_csv(csv_path)
    all_image_paths.extend([os.path.abspath(img.strip()) for img in df['img_path']])

    all_labels.extend(df[['hotend_class', 'z_offset_class', 'feed_rate_class', 'flow_rate_class']].values)
    all_nozzle_coords.extend(df[['nozzle_tip_x', 'nozzle_tip_y']].values)

image_train, image_temp, labels_train, labels_temp, nozzle_coords_train, nozzle_coords_temp = train_test_split(
    all_image_paths, all_labels, all_nozzle_coords, test_size=0.30, random_state=42
)

image_val, image_test, labels_val, labels_test, nozzle_coords_val, nozzle_coords_test = train_test_split(
    image_temp, labels_temp, nozzle_coords_temp, test_size=0.40, random_state=42
)

mean_std_calculator = CalculateMeanStd(image_paths=image_train)
channels_mean, channels_std = mean_std_calculator()

transform_pipeline = TransformPipeline([
    RandomRotate(),
    RandomPerspectiveTransform(),
    CenteredCrop(crop_size=100, output_size=320),
    RandomCrop(output_size=224),
    HorizontalFlipColorJitter(),
    NormalizeChannels(image_train)
])

val_transform_pipeline = TransformPipeline([
    CenteredCrop(crop_size=100, output_size=320),
    RandomCrop(output_size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=channels_mean, std=channels_std)
])

test_transform_pipeline = val_transform_pipeline


class PrintAnomalyClassifier():
    def __init__(self):
        self.train_dataset = CustomDataset(image_train, labels_train, nozzle_coords=nozzle_coords_train,
                                           transform=transform_pipeline)
        self.val_dataset = CustomDataset(image_val, labels_val, nozzle_coords=nozzle_coords_val,
                                         transform=val_transform_pipeline)
        self.test_dataset = CustomDataset(image_test, labels_test, nozzle_coords=nozzle_coords_test,
                                          transform=test_transform_pipeline)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False, drop_last=False)
        self.model = RosNet(in_channels=3, out_channels=64)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_epochs = 30
        self.early_stopping_patience = 10
        self.epochs_without_improvement = 0
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def validate_model(self, val_loader, loss_fn):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output1, output2, output3, output4 = self.model(images)
                loss1 = loss_fn(output1, labels[:, 0].long())  # hotend
                loss2 = loss_fn(output2, labels[:, 1].long())  # z_offset
                loss3 = loss_fn(output3, labels[:, 2].long())  # feed_rate
                loss4 = loss_fn(output4, labels[:, 3].long())  # flow_rate_
                val_loss += loss1.item() + loss2.item() + loss3.item() + loss4.item()

        avg_loss = val_loss / len(val_loader)
        return avg_loss

    def train_loop(self):
        print("Starting or resuming training")
        self.model.to(self.device)

        # Load saved state dictionary if it exists
        checkpoint_path = '/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/final_current_unet.pth'
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print("No checkpoint found. Starting training from scratch.")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            print("Epoch num:", epoch)
            self.model.train()
            running_loss = 0.0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()
                output1, output2, output3, output4 = self.model(images)

                loss1 = self.loss_fn(output1, labels[:, 0].long())  # hotend
                loss2 = self.loss_fn(output2, labels[:, 1].long())  # z_offset
                loss3 = self.loss_fn(output3, labels[:, 2].long())  # feed_rate
                loss4 = self.loss_fn(output4, labels[:, 3].long())  # flow_rate_

                total_loss = loss1 + loss2 + loss3 + loss4
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()

            print("Evaluating early stopping")
            avg_val_loss = self.validate_model(self.val_loader, self.loss_fn)

            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'final_best_unet.pth')
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                torch.save(self.model.state_dict(), 'final_current_unet.pth')
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        print("Training finished")

Classifier = PrintAnomalyClassifier()
Classifier.train_loop()
