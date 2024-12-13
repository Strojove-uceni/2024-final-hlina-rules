
image_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/caxton_dataset_big/print0/image-6.jpg"


import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from UnetModel import RosNet
# Helper function to overlay mask on image
def overlay_mask(image, mask, alpha=0.5, cmap="jet"):
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/final_current_rosnet.pth"

output_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/Attention3_mask_visualization.png"

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Load the model
model = RosNet(in_channels=3, out_channels=64)  # Adjust in_channels and out_channels if necessary
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Enable mask retrieval for the third attention module
model.attention3.retrieve_mask = True

# Forward pass and retrieve mask
with torch.no_grad():
    x = model.double_conv(input_tensor)
    x_down1, x_pooled1 = model.down1(x)
    b1 = model.bottleneck1(x_pooled1)
    att1_output = model.attention1(b1)
    x_up1 = model.up1(att1_output, x_down1)

    x_down2, x_pooled2 = model.down2(x_up1)
    b2 = model.bottleneck2(x_pooled2)
    att2_output = model.attention2(b2)
    x_up2 = model.up2(att2_output, x_down2)

    x_down3, x_pooled3 = model.down3(x_up2)
    b3 = model.bottleneck3(x_pooled3)
    att3_output, mask = model.attention3(b3)  # Retrieve mask here

# Process the mask for visualization
mask = mask[0, 0].cpu().numpy()
mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # Normalize mask
mask_resized = np.array(Image.fromarray(mask).resize(image.size, resample=Image.BILINEAR))
mask_smoothed = gaussian_filter(mask_resized, sigma=5)  # Optional smoothing

# Visualize and save
plt.figure(figsize=(10, 10))
overlay_mask(np.array(image), mask_smoothed)
plt.savefig(output_path)
plt.close()

print(f"Attention mask for stage 3 saved at {output_path}.")

