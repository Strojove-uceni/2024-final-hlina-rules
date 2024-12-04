import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ImagePreprocessing import CustomDataset, TransformPipeline
from torch.utils.data import DataLoader
from ImagePreprocessing import RandomRotate, RandomPerspectiveTransform, CenteredCrop, RandomCrop, HorizontalFlipColorJitter
from ImagePreprocessing import CalculateMeanStd, NormalizeChannels
import torchvision.transforms as transforms
from Class_classification_Model import ResidualAttentionNetwork
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import time


DATASET_DIR = '/Users/lydierosenkrancova/Desktop/projects/SU2_CambridgeData/SU2_cambridge_data_FirstAttempt_multifiles_load/caxton_dataset'

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
    img_path = all_image_paths[0]
    print(img_path)


    all_labels.extend(df[['hotend_class', 'z_offset_class', 'feed_rate_class', 'flow_rate_class']].values)
    all_nozzle_coords.extend(df[['nozzle_tip_x', 'nozzle_tip_y']].values)


image_train, image_temp, labels_train, labels_temp, nozzle_coords_train, nozzle_coords_temp = train_test_split(
    all_image_paths, all_labels, all_nozzle_coords, test_size=0.999, random_state=42
)

image_val, image_tempor, labels_val, labels_tempor, nozzle_coords_val, nozzle_coords_tempor = train_test_split(
    image_temp, labels_temp, nozzle_coords_temp, test_size=0.999, random_state=42
)

image_test, image_redund, labels_test, labels_redund, nozzle_coords_test, nozzle_coords_redund = train_test_split(
    image_tempor, labels_tempor, nozzle_coords_tempor, test_size=0.999, random_state=42
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
        self.train_dataset = CustomDataset(image_train, labels_train, nozzle_coords=nozzle_coords_train, transform=transform_pipeline)
        self.val_dataset = CustomDataset(image_val, labels_val,nozzle_coords=nozzle_coords_val, transform=val_transform_pipeline)
        self.test_dataset = CustomDataset(image_test, labels_test, nozzle_coords=nozzle_coords_test, transform=test_transform_pipeline)
        self.train_loader = DataLoader(self.train_dataset, batch_size=3, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=3, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=3, shuffle=False, drop_last=False)
        self.model = ResidualAttentionNetwork()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_epochs = 2
        self.early_stopping_patience = 10
        self.epochs_without_improvement = 0
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def validate_model(self, val_loader, loss_fn):
        self.model.eval()
        val_loss = 0.0
        hotend_predictions = []
        hotend_labels = []
        z_offset_predictions = []
        z_offset_labels = []
        feed_rate_predictions = []
        feed_rate_labels = []
        flow_rate_labels = []
        flow_rate_predictions = []

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


                hotend_prediction = torch.argmax(output1, dim=1).cpu().numpy()
                hotend_label = labels[:, 0].long().cpu().numpy()
                hotend_predictions.append(hotend_prediction)
                hotend_labels.append(hotend_label)

                z_offset_prediction = torch.argmax(output2, dim=1).cpu().numpy()
                z_offset_label = labels[:, 1].long().cpu().numpy()
                z_offset_predictions.append(z_offset_prediction)
                z_offset_labels.append(z_offset_label)

                feed_rate_prediction = torch.argmax(output3, dim=1).cpu().numpy()
                feed_rate_label = labels[:, 2].long().cpu().numpy()
                feed_rate_predictions.append(feed_rate_prediction)
                feed_rate_labels.append(feed_rate_label)

                flow_rate_prediction = torch.argmax(output4, dim=1).cpu().numpy()
                flow_rate_label = labels[:, 3].long().cpu().numpy()
                flow_rate_predictions.append(flow_rate_prediction)
                flow_rate_labels.append(flow_rate_label)

            hotend_predictions = np.concatenate(hotend_predictions)
            hotend_labels = np.concatenate(hotend_labels)
            z_offset_predictions = np.concatenate(z_offset_predictions)
            z_offset_labels = np.concatenate(z_offset_labels)
            feed_rate_predictions = np.concatenate(feed_rate_predictions)
            feed_rate_labels = np.concatenate(feed_rate_labels)
            flow_rate_predictions = np.concatenate(flow_rate_predictions)
            flow_rate_labels = np.concatenate(flow_rate_labels)

        avg_loss = val_loss / len(val_loader)
        return avg_loss, hotend_predictions, hotend_labels, z_offset_predictions, z_offset_labels, feed_rate_predictions, feed_rate_labels, flow_rate_predictions, flow_rate_labels

    def train_loop(self):
        print("starting training")
        self.model.to(self.device)
        best_val_loss = float('inf')
        total_start_time = time.time()
        for epoch in range(self.num_epochs):
            print("epoch num:", epoch)
            self.model.train()
            running_loss = 0.0
            train_hotend_predictions = []
            train_hotend_labels = []
            train_z_offset_predictions = []
            train_z_offset_labels = []
            train_feed_rate_predictions = []
            train_feed_rate_labels = []
            train_flow_rate_labels = []
            train_flow_rate_predictions = []

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()
                output1, output2, output3, output4 = self.model(images)

                loss1 = self.loss_fn(output1, labels[:, 0].long()) #hotend
                loss2 = self.loss_fn(output2, labels[:, 1].long()) #z_offset
                loss3 = self.loss_fn(output3, labels[:, 2].long()) #feed_rate
                loss4 = self.loss_fn(output4, labels[:, 3].long()) #flow_rate_

                total_loss = loss1 + loss2 + loss3 + loss4
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item()

                hotend_prediction = torch.argmax(output1, dim=1).cpu().numpy()
                hotend_label = labels[:, 0].long().cpu().numpy()
                train_hotend_predictions.append(hotend_prediction)
                train_hotend_labels.append(hotend_label)

                z_offset_prediction = torch.argmax(output2, dim=1).cpu().numpy()
                z_offset_label = labels[:, 1].long().cpu().numpy()
                train_z_offset_predictions.append(z_offset_prediction)
                train_z_offset_labels.append(z_offset_label)

                feed_rate_prediction = torch.argmax(output3, dim=1).cpu().numpy()
                feed_rate_label = labels[:, 2].long().cpu().numpy()
                train_feed_rate_predictions.append(feed_rate_prediction)
                train_feed_rate_labels.append(feed_rate_label)

                flow_rate_prediction = torch.argmax(output4, dim=1).cpu().numpy()
                flow_rate_label = labels[:, 3].long().cpu().numpy()
                train_flow_rate_predictions.append(flow_rate_prediction)
                train_flow_rate_labels.append(flow_rate_label)

            train_hotend_predictions = np.concatenate(train_hotend_predictions)
            train_hotend_labels = np.concatenate(train_hotend_labels)
            train_z_offset_predictions = np.concatenate(train_z_offset_predictions)
            train_z_offset_labels = np.concatenate(train_z_offset_labels)
            train_feed_rate_predictions = np.concatenate(train_feed_rate_predictions)
            train_feed_rate_labels = np.concatenate(train_feed_rate_labels)
            train_flow_rate_predictions = np.concatenate(train_flow_rate_predictions)
            train_flow_rate_labels = np.concatenate(train_flow_rate_labels)
            print("Evaluating early stopping")
            avg_val_loss, val_hotend_predictions, val_hotend_labels, val_z_offset_predictions, val_z_offset_labels, val_feed_rate_predictions, val_feed_rate_labels, val_flow_rate_predictions, val_flow_rate_labels= self.validate_model(self.val_loader, self.loss_fn)

            self.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        print("training finished")
        return train_hotend_predictions, train_hotend_labels, train_z_offset_predictions, train_z_offset_labels, train_feed_rate_predictions, train_feed_rate_labels, train_flow_rate_predictions, train_flow_rate_labels, avg_val_loss, val_hotend_predictions, val_hotend_labels, val_z_offset_predictions, val_z_offset_labels, val_feed_rate_predictions, val_feed_rate_labels, val_flow_rate_predictions, val_flow_rate_labels, total_start_time

    def compute_metrics(self, labels, predictions):
        print("computing metrics")
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        cf = confusion_matrix(labels, predictions)
        metrics_dict = {
            'accuracy:': accuracy,
            'precision:': precision,
            'recall:': recall,
            'f1:': f1,
            'confusion_matrix:': cf
            }

        return metrics_dict

    def test_model(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        hotend_predictions = []
        hotend_labels = []
        z_offset_predictions = []
        z_offset_labels = []
        feed_rate_predictions = []
        feed_rate_labels = []
        flow_rate_labels = []
        flow_rate_predictions = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                output1, output2, output3, output4 = self.model(images)

                hotend_prediction = torch.argmax(output1, dim=1).cpu().numpy()
                hotend_label = labels[:, 0].long().cpu().numpy()
                hotend_predictions.append(hotend_prediction)
                hotend_labels.append(hotend_label)

                z_offset_prediction = torch.argmax(output2, dim=1).cpu().numpy()
                z_offset_label = labels[:, 1].long().cpu().numpy()
                z_offset_predictions.append(z_offset_prediction)
                z_offset_labels.append(z_offset_label)

                feed_rate_prediction = torch.argmax(output3, dim=1).cpu().numpy()
                feed_rate_label = labels[:, 2].long().cpu().numpy()
                feed_rate_predictions.append(feed_rate_prediction)
                feed_rate_labels.append(feed_rate_label)

                flow_rate_prediction = torch.argmax(output4, dim=1).cpu().numpy()
                flow_rate_label = labels[:, 3].long().cpu().numpy()
                flow_rate_predictions.append(flow_rate_prediction)
                flow_rate_labels.append(flow_rate_label)

            hotend_predictions = np.concatenate(hotend_predictions)
            hotend_labels = np.concatenate(hotend_labels)
            z_offset_predictions = np.concatenate(z_offset_predictions)
            z_offset_labels = np.concatenate(z_offset_labels)
            feed_rate_predictions = np.concatenate(feed_rate_predictions)
            feed_rate_labels = np.concatenate(feed_rate_labels)
            flow_rate_predictions = np.concatenate(flow_rate_predictions)
            flow_rate_labels = np.concatenate(flow_rate_labels)

        return hotend_predictions, hotend_labels, z_offset_predictions, z_offset_labels, feed_rate_predictions, feed_rate_labels, flow_rate_predictions, flow_rate_labels

    def evaluate_model(self):
        train_hotend_predictions, train_hotend_labels, train_z_offset_predictions, train_z_offset_labels, train_feed_rate_predictions, train_feed_rate_labels, train_flow_rate_predictions, train_flow_rate_labels, avg_val_loss, val_hotend_predictions, val_hotend_labels, val_z_offset_predictions, val_z_offset_labels, val_feed_rate_predictions, val_feed_rate_labels, val_flow_rate_predictions, val_flow_rate_labels, total_start_time = self.train_loop()
        test_hotend_predictions, test_hotend_labels, test_z_offset_predictions, test_z_offset_labels, test_feed_rate_predictions, test_feed_rate_labels, test_flow_rate_predictions, test_flow_rate_labels = self.test_model()

        train_hotend_metrics = self.compute_metrics(train_hotend_labels, train_hotend_predictions)
        train_z_offset_metrics = self.compute_metrics(train_z_offset_labels, train_z_offset_predictions)
        train_feed_rate_metrics = self.compute_metrics(train_feed_rate_labels, train_feed_rate_predictions)
        train_flow_rate_metrics = self.compute_metrics(train_flow_rate_labels, train_flow_rate_predictions)

        val_hotend_metrics = self.compute_metrics(val_hotend_labels, val_hotend_predictions)
        val_z_offset_metrics = self.compute_metrics(val_z_offset_labels, val_z_offset_predictions)
        val_feed_rate_metrics = self.compute_metrics(val_feed_rate_labels, val_feed_rate_predictions)
        val_flow_rate_metrics = self.compute_metrics(val_flow_rate_labels, val_flow_rate_predictions)

        test_hotend_metrics = self.compute_metrics(test_hotend_labels, test_hotend_predictions)
        test_z_offset_metrics = self.compute_metrics(test_z_offset_labels, test_z_offset_predictions)
        test_feed_rate_metrics = self.compute_metrics(test_feed_rate_labels, test_feed_rate_predictions)
        test_flow_rate_metrics = self.compute_metrics(test_flow_rate_labels, test_flow_rate_predictions)

        train_metrics = {
            'train_hotend_metrics:': train_hotend_metrics,
            'train_z_offset_metrics:': train_z_offset_metrics,
            'train_feed_rate_metrics:': train_feed_rate_metrics,
            'train_flow_rate_metrics:': train_flow_rate_metrics
            }

        val_metrics = {
            'val_hotend_metrics': val_hotend_metrics,
            'val_z_offset_metrics:': val_z_offset_metrics,
            'val_feed_rate_metrics': val_feed_rate_metrics,
            'val_flow_rate_metrics:': val_flow_rate_metrics
        }

        test_metrics = {
            'test_hotend_metrics:': test_hotend_metrics,
            'test_z_offset_metrics:': test_z_offset_metrics,
            'test_feed_rate_metrics:': test_feed_rate_metrics,
            'test_flow_rate_metrics': test_flow_rate_metrics
        }

        with open('train_metrics.pkl', 'wb') as f:
            pickle.dump(train_metrics, f)

        with open('val_metrics.pkl', 'wb') as f:
            pickle.dump(val_metrics, f)

        with open('test_metrics.pkl', 'wb') as f:
            pickle.dump(test_metrics, f)

        print("Metrics saved to pickle files.")
        end_time = time.time()
        return train_metrics, val_metrics, test_metrics, end_time, total_start_time

    def run(self):
        train_metrics, val_metrics, test_metrics, end_time, total_start_time = self.evaluate_model()
        time_dict = { 'start_time:': total_start_time,
                      'end_time:': end_time

        }
        with open('time_elapsed.pkl', 'wb') as f:
            pickle.dump(time_dict, f)
        print("Training, validation, and testing completed!")
        
Classifier = PrintAnomalyClassifier()
Classifier.run()

