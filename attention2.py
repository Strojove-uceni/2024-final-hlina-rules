import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from Models_components import AttentionModule_stage3
from Model import ResidualAttentionNetwork


def overlay_mask(image, mask, alpha=0.5, cmap="jet"):
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/s_best_articlenet.pth"
image_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/caxton_dataset/print15/image-48.jpg"
output_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/ArticleNet3_final_attention3_mask_improved.png"

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

model = ResidualAttentionNetwork()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

model.to(device)
model.eval()

model.attention3.retrieve_mask = True

with torch.no_grad():
    x = model.initial_conv(input_tensor)
    x = model.pool(x)
    x = model.residual0(x)
    x = model.attention1(x)  
    x = model.residual1(x)
    x = model.attention2(x)  
    x = model.residual2(x)
    trunk_output, mask = model.attention3(x)  

mask = mask[0, 0].cpu().numpy()
mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  


mask_resized = np.array(Image.fromarray(mask).resize(image.size, resample=Image.BILINEAR))

# Apply Gaussian smoothing to the mask
mask_smoothed = gaussian_filter(mask_resized, sigma=5)


plt.figure(figsize=(10, 10))
overlay_mask(np.array(image), mask_smoothed)
plt.savefig(output_path)
plt.close()

print(f"Improved attention mask for stage 3 saved at {output_path}.")

