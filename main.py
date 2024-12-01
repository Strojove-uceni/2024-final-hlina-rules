import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ImagePreprocessing import CustomDataset, TransformPipeline
from torch.utils.data import DataLoader
from ImagePreprocessing import RandomRotate, RandomPerspectiveTransform, CenteredCrop, RandomCrop, HorizontalFlipColorJitter
from ImagePreprocessing import CalculateMeanStd, NormalizeChannels
import torchvision.transforms as transforms
from Model import ResidualAttentionNetwork
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
DATASET_DIR = 'caxton_dataset'
PRINT_DIR = os.path.join(DATASET_DIR, 'print0')  # Folder containing images and CSV
LABELS_PATH = os.path.join(PRINT_DIR, 'print_log_full.csv')
df = pd.read_csv(LABELS_PATH)
image_paths = df['img_path'].tolist()


labels = df[['flow_rate', 'feed_rate', 'z_offset', 'target_hotend']].values
nozzle_coords = df[['nozzle_tip_x', 'nozzle_tip_y']].values


image_train, image_temp, labels_train, labels_temp, nozzle_coords_train, nozzle_coords_temp = train_test_split(
    image_paths, labels, nozzle_coords, test_size=0.999, random_state=42
)

image_val, image_test, labels_val, labels_test, nozzle_coords_val, nozzle_coords_test = train_test_split(
    image_temp, labels_temp, nozzle_coords_temp, test_size=0.999, random_state=42
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

train_dataset = CustomDataset(image_train, labels_train, transform=transform_pipeline)
val_dataset = CustomDataset(image_val, labels_val, transform=val_transform_pipeline)
test_dataset = CustomDataset(image_test, labels_test, transform=test_transform_pipeline)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)



def train_loop():

    model = ResidualAttentionNetwork()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    num_epochs = 1
    early_stopping_patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)
    val_loss_history = []
    val_mae, val_mse, val_rmse = 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu').float()

            optimizer.zero_grad()
            output1, output2, output3, output4 = model(images)

            loss1 = loss_fn(output1, labels[:, 0].unsqueeze(1))
            loss2 = loss_fn(output2, labels[:, 1].unsqueeze(1))
            loss3 = loss_fn(output3, labels[:, 2].unsqueeze(1))
            loss4 = loss_fn(output4, labels[:, 3].unsqueeze(1))

            total_loss = loss1 + loss2 + loss3 + loss4
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_outputs = [[] for _ in range(4)]
        all_labels = [[] for _ in range(4)]
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu').float()
                output1, output2, output3, output4 = model(images)


                loss1 = loss_fn(output1, labels[:, 0].unsqueeze(1))
                loss2 = loss_fn(output2, labels[:, 1].unsqueeze(1))
                loss3 = loss_fn(output3, labels[:, 2].unsqueeze(1))
                loss4 = loss_fn(output4, labels[:, 3].unsqueeze(1))

                val_loss += (loss1 + loss2 + loss3 + loss4).item()

                all_outputs[0].append(output1.cpu().numpy())
                all_outputs[1].append(output2.cpu().numpy())
                all_outputs[2].append(output3.cpu().numpy())
                all_outputs[3].append(output4.cpu().numpy())
                all_labels[0].append(labels[:, 0].cpu().numpy())
                all_labels[1].append(labels[:, 1].cpu().numpy())
                all_labels[2].append(labels[:, 2].cpu().numpy())
                all_labels[3].append(labels[:, 3].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        for i in range(4):
            all_outputs[i] = np.concatenate(all_outputs[i], axis=0)
            all_labels[i] = np.concatenate(all_labels[i], axis=0)


        for i in range(4):
            if i < len(all_outputs) and i < len(all_labels):
                val_mae = np.mean(np.abs(all_outputs[i] - all_labels[i]))
                val_mse = np.mean((all_outputs[i] - all_labels[i]) ** 2)
                val_rmse = np.sqrt(val_mse)



        scheduler.step(avg_val_loss)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    val_metrics_dict = {'val_mae': val_mae,
                        'val_mse': val_mse,
                        'val_rmse': val_rmse
                        }
    with open('validation_metrics.pkl', "wb") as f:
        pickle.dump(val_metrics_dict, f)


    with open("val_loss_history.txt", "w") as f:
        for loss in val_loss_history:
            f.write(f"{loss}\n")

    model.eval()
    test_mae = 0.0
    test_mse = 0.0
    test_rmse = 0.0
    test_outputs = [[] for _ in range(4)]
    test_labels = [[] for _ in range(4)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu').float()
            output1, output2, output3, output4 = model(images)

            test_outputs[0].append(output1.cpu().numpy())
            test_outputs[1].append(output2.cpu().numpy())
            test_outputs[2].append(output3.cpu().numpy())
            test_outputs[3].append(output4.cpu().numpy())
            test_labels[0].append(labels[:, 0].cpu().numpy())
            test_labels[1].append(labels[:, 1].cpu().numpy())
            test_labels[2].append(labels[:, 2].cpu().numpy())
            test_labels[3].append(labels[:, 3].cpu().numpy())

            for i in range(4):
                test_outputs[i] = np.concatenate(test_outputs[i], axis=0)
                test_labels[i] = np.concatenate(test_labels[i], axis=0)

            for i in range(4):
                if i < len(test_outputs) and i < len(test_labels):
                    test_mae = np.mean(np.abs(test_outputs[i] - test_labels[i]))
                    test_mse = np.mean((test_outputs[i] - test_labels[i]) ** 2)
                    test_rmse = np.sqrt(test_mse)

    test_metrics_dict = {'test_mae': test_mae,
                         'test_mse': test_mse,
                         'test_rmse': test_rmse
                        }
    with open('test_metrics.pkl', "wb") as f:
        pickle.dump(test_metrics_dict, f)

train_loop()
