import os
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from ImagePreprocessing import CustomDataset, TransformPipeline
from Class_classification_Model import ResidualAttentionNetwork
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
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

#this code was used because during training we ran out of time (maximum walltime on cluster is 24 hours)
#because of this I had to load the trained model and evaluate it in separate run
DATASET_DIR = '/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/caxton_dataset'
MODEL_PATH = '/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/s_best_articlenet.pth'

# Prepare test dataset
print_dirs = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) if d.startswith('print')]

all_image_paths = []
all_labels = []
all_nozzle_coords = []

print("Preparing directories")
for print_dir in print_dirs:
    csv_path = os.path.join(print_dir, 'print_log_filtered_classification3.csv')
    if not os.path.exists(csv_path):
        print(f"Skipping {print_dir}: Missing 'print_log_filtered_classification3.csv'")
        continue

    df = pd.read_csv(csv_path)
    all_image_paths.extend([os.path.abspath(img.strip()) for img in df['img_path']])
    all_labels.extend(df[['hotend_class', 'z_offset_class', 'feed_rate_class', 'flow_rate_class']].values)
    all_nozzle_coords.extend(df[['nozzle_tip_x', 'nozzle_tip_y']].values)

image_train, image_temp, labels_train, labels_temp, nozzle_coords_train, nozzle_coords_temp = train_test_split(
    all_image_paths, all_labels, all_nozzle_coords, test_size=0.7, random_state=42
)

image_val, image_test, labels_val, labels_test, nozzle_coords_val, nozzle_coords_test = train_test_split(
    image_temp, labels_temp, nozzle_coords_temp, test_size=0.4, random_state=42
)

mean_std_calculator = CalculateMeanStd(image_paths=image_train)
channels_mean, channels_std = mean_std_calculator()

test_transform_pipeline = TransformPipeline([
    CenteredCrop(crop_size=100, output_size=320),
    RandomCrop(output_size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=channels_mean, std=channels_std)
])

test_dataset = CustomDataset(image_test, labels_test, nozzle_coords=nozzle_coords_test, transform=test_transform_pipeline)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResidualAttentionNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


hotend_predictions, hotend_labels = [], []
z_offset_predictions, z_offset_labels = [], []
feed_rate_predictions, feed_rate_labels = [], []
flow_rate_predictions, flow_rate_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        output1, output2, output3, output4 = model(images)

        hotend_predictions.append(torch.argmax(output1, dim=1).cpu().numpy())
        hotend_labels.append(labels[:, 0].long().cpu().numpy())

        z_offset_predictions.append(torch.argmax(output2, dim=1).cpu().numpy())
        z_offset_labels.append(labels[:, 1].long().cpu().numpy())

        feed_rate_predictions.append(torch.argmax(output3, dim=1).cpu().numpy())
        feed_rate_labels.append(labels[:, 2].long().cpu().numpy())

        flow_rate_predictions.append(torch.argmax(output4, dim=1).cpu().numpy())
        flow_rate_labels.append(labels[:, 3].long().cpu().numpy())


hotend_predictions = np.concatenate(hotend_predictions)
hotend_labels = np.concatenate(hotend_labels)
z_offset_predictions = np.concatenate(z_offset_predictions)
z_offset_labels = np.concatenate(z_offset_labels)
feed_rate_predictions = np.concatenate(feed_rate_predictions)
feed_rate_labels = np.concatenate(feed_rate_labels)
flow_rate_predictions = np.concatenate(flow_rate_predictions)
flow_rate_labels = np.concatenate(flow_rate_labels)


def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    cf = confusion_matrix(labels, predictions)
    return {
        'accuracy:': accuracy,
        'precision:': precision,
        'recall:': recall,
        'f1:': f1,
        'confusion_matrix:': cf
    }

test_metrics = {
    'test_hotend_metrics': compute_metrics(hotend_labels, hotend_predictions),
    'test_z_offset_metrics': compute_metrics(z_offset_labels, z_offset_predictions),
    'test_feed_rate_metrics': compute_metrics(feed_rate_labels, feed_rate_predictions),
    'test_flow_rate_metrics': compute_metrics(flow_rate_labels, flow_rate_predictions)
}

# Save metrics and predictions
with open('final_articlenet_test_metricsfrfrf.pkl', 'wb') as f:
    pickle.dump(test_metrics, f)

test_predslabels = {
    'test_hotend_predictions': hotend_predictions,
    'test_hotend_labels': hotend_labels,
    'test_z_offset_predictions': z_offset_predictions,
    'test_z_offset_labels': z_offset_labels,
    'test_feed_rate_predictions': feed_rate_predictions,
    'test_feed_rate_labels': feed
}


with open('final_article_test_predslabelsfrfrf.pkl', 'wb') as f:
    pickle.dump(test_metrics, f)
