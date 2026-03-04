import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch.nn.functional as F

## For setting current directory and working with files
import os, sys, platform

wd=os.getcwd()
print(wd)

# Setting path based on current platform, so I can import my libraries
windowsPath = r"M:\projects"
linuxPath = "/home/s2807393/RDS/projects"

# Choose path based on system
if platform.system() == "Windows":
    sys.path.append(windowsPath)
    PROJECT_FOLDER = windowsPath
else: # Linux
    sys.path.append(linuxPath)
    PROJECT_FOLDER = linuxPath

# My utility functions
import myutils
import dino_utils

# %%
# --- CONFIG ---
# Path to your RAW test images (not the features, the actual .png files)
TEST_IMG_DIR = "/home/s2807393/RDS/projects/data/MAVIC_C_2025/test"
MODEL_NAME = "vits_plus" #"convnext_l" # "vitl_sat" #"vith_plus#

OUTPUT_PATH = "/home/s2807393/RDS/projects/dino_tests/dino_" + MODEL_NAME + "_unicornv2_test_features_TTA.pt"


# %% Select device (GPU or CPU)
if torch.cuda.is_available():
    device = myutils.get_freer_gpu()
else:
    device = "cpu"
print(f"Using {device} device")

# Load Model
model, _ = dino_utils.load_dino_model(model_type=MODEL_NAME)
model.to(device)
model.eval()

# --- TTA TRANSFORM ---
# We defined 5 views:
# 1. Standard
# 2. Horizontal Flip
# 3. Slight Zoom (110%) -> Center Crop
# 4. Brighter
# 5. Darker
tta_transform = dino_utils.make_tta_transforms(model_type=MODEL_NAME, resize_size=256)

# %% # Custom Dataset that returns List of 5 images instead of 1
class TTADataset(Dataset):
    def __init__(self, root,):
        self.root = root
        self.imgs = sorted([f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg'))])
        self.transforms = tta_transform


    def __getitem__(self, idx):
        path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(path).convert("RGB")

        # Apply all transforms and stack them: [N_views, 3, 256, 256]
        aug_imgs = torch.stack([t(img) for t in self.transforms])
        return aug_imgs, 0  # Dummy label

    def __len__(self):
        return len(self.imgs)


def extract_features_tta():
    dataset = TTADataset(TEST_IMG_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_features = []

    print(f"Extracting TTA features for {len(dataset)} images...")
    ii  = 0
    total_images = len(dataset)

    with torch.no_grad():
        for batch_imgs, _ in loader:
            ii = ii + 1
            # batch_imgs shape: [Batch, Views(5), Channels, H, W]
            b, v, c, h, w = batch_imgs.shape

            # Flatten to: [Batch*Views, C, H, W]
            flat_imgs = batch_imgs.view(b * v, c, h, w).to(device)

            # Extract features
            features = model(flat_imgs)  # [Batch*Views, Dim]

            # Reshape back to [Batch, Views, Dim]
            features = features.view(b, v, -1)

            # MEAN POOLING over the Views dimension
            # This averages the features of the 3 or 5 augmentations
            avg_features = features.mean(dim=1)

            all_features.append(avg_features.cpu())

            processed_count = ii * loader.batch_size
            if ii % 20 == 0:
                pct_done = processed_count/total_images * 100
                print(f"Batch {ii+1}: Processed {processed_count} out of {total_images} images - {pct_done}% done...")


    final_features = torch.cat(all_features, dim=0)

    # Save in the same format as before so your other scripts work
    torch.save({'features': final_features, 'labels': torch.zeros(len(dataset))}, OUTPUT_PATH)
    print(f"Saved TTA Features to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_features_tta()