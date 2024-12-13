import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.models import resnet50
from Model import CustomResNet

#this code was used to visualize the attention mask of the model
def overlay_mask(image, mask, alpha=0.5, cmap="jet"):
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()

features = []

def hook_fn(module, input, output):
    features.append(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/short_best_resnet.pth"
image_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/caxton_dataset/print15/image-48.jpg"
output_path = "/mnt/lustre/helios-home/rosenlyd/2024-final-hlina-rules/2024-final-hlina-rules/CustomResNetfinal_attention_mask.png"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

base_resnet = resnet50(pretrained=False)  
model = CustomResNet(base_resnet)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


layer_to_hook = model.base_model.layer4  
hook = layer_to_hook.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(input_tensor) 


hook.remove()

feature_map = features[0][0]  #
attention_mask = torch.mean(feature_map, dim=0).cpu().numpy()  
attention_mask = (attention_mask - np.min(attention_mask)) / (np.max(attention_mask) - np.min(attention_mask))  # Normalize

mask_resized = np.array(Image.fromarray(attention_mask).resize(image.size, resample=Image.BILINEAR))

# Apply Gaussian smoothing to the mask
mask_smoothed = gaussian_filter(mask_resized, sigma=5)


plt.figure(figsize=(10, 10))
overlay_mask(np.array(image), mask_smoothed)
plt.savefig(output_path)
plt.close()

print(f"Attention mask saved at {output_path}.")
