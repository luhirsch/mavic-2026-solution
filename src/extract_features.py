"""
Script: extract_features.py
Purpose: Extracts feature embeddings using the DINOv3 model.
         Processes the SAR training dataset and processes the SAR test dataset using
         Test-Time Augmentation (TTA) by averaging features across 5 augmented views.
         These features are saved to disk and will be used for training the
         Mahalanobis OOD detector in inference.py.

Outputs:
    - Training features: Features for the SAR training dataset (SAVE_PATH_TRAIN)
    - TTA Test features: TTA Features for the SAR test dataset (SAVE_PATH_TEST)
"""


import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import yaml
from pathlib import Path

import model_utils

# %%
# ==============================================
#  DIRECTORY SETTINGS (obtained from config.yaml)
# ==============================================

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# DINOv3 weights - Please update with your paths
DINO_WEIGHTS = Path(cfg["paths"]["dino_weights_vitl_sat"]) # Use SAT model for OOD Detection
DINO_REPO = Path(cfg["paths"]["dino_repo"])

# Data - Put MAVIC-C 2025 data in the "data" folder or update with your paths
TRAIN_DATA_SAR = Path(cfg["paths"]["train_sar"])
TEST_DATA_SAR = Path(cfg["paths"]["test"]) # This folder is obtained after running "organize_mavic_val_into_folders"

# Outputs (feature embeddings)
OUTPUT_DIR = "./output"
SAVE_PATH_TRAIN = os.path.join(OUTPUT_DIR, "train_features_vitl_sat.pt")
SAVE_PATH_TEST = os.path.join(OUTPUT_DIR, "test_features_TTA.pt")


# %%
# ==========================================
#        EXTRACTION SETTINGS
# ==========================================

# Load Model
MODEL_TYPE = "dinov3_vitl16"
IMAGE_SIZE_MAHA = 256
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% # Custom Dataset that returns List of 5 images instead of 1
# ==========================================
#        DATASET
# ==========================================

class TTADataset(Dataset):
    """
    Custom Dataset that returns a stack of 5 augmented views per image.
    Returns the filename instead of a dummy label for submission tracking.
    """

    def __init__(self, root, resize_size=128):
        self.root = root
        self.imgs = sorted([f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        self.transforms = model_utils.make_tta_transforms(resize_size=resize_size)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        path = os.path.join(self.root, img_name)
        img = Image.open(path).convert("RGB")

        # Apply all transforms and stack them: [N_views, 3, H, W]
        aug_imgs = torch.stack([t(img) for t in self.transforms])

        # Return the filename instead of the dummy label
        return aug_imgs, img_name

    def __len__(self):
        return len(self.imgs)

# %%
# %%
# ==========================================
#        EXTRACTION FUNCTIONS
# ==========================================

def extract_train_features(model):
    """Extracts standard single-view features for the training dataset."""
    print(f"\n--- Preparing Train Dataset ---")
    train_transform = model_utils.make_dino_transform(resize_size=IMAGE_SIZE_MAHA)
    train_ds = datasets.ImageFolder(TRAIN_DATA_SAR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"--- Extracting Train Features ({len(train_ds)} images) ---")
    train_features = []
    train_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            feats = model(images)

            train_features.append(feats.cpu())
            train_labels.append(labels.cpu())

            if i % 20 == 0:
                print(f"Train Extraction: Batch {i}/{len(train_loader)}")

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    torch.save({"features": train_features, "labels": train_labels}, SAVE_PATH_TRAIN)
    print(f">>> Saved train features to {SAVE_PATH_TRAIN} | Shape: {train_features.shape}")


def extract_test_features_tta(model):
    """Extracts Multi-view (TTA) features for the test dataset and averages them."""
    print(f"\n--- Preparing Test Dataset (TTA) ---")
    test_ds = TTADataset(TEST_DATA_SAR, resize_size=IMAGE_SIZE_MAHA)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"--- Extracting Test Features with TTA ({len(test_ds)} images) ---")
    test_features = []
    test_filenames = []

    with torch.no_grad():
        for i, (images, filenames) in enumerate(test_loader):
            # images shape: [B, 5, 3, H, W]
            b, n_views, c, h, w = images.shape

            # Reshape to [B*5, 3, H, W] to pass through the model
            images = images.view(-1, c, h, w).to(DEVICE)

            feats = model(images)  # Output shape: [B*5, D]

            # Reshape back to [B, 5, D] and average across the 5 views
            feats = feats.view(b, n_views, -1)
            averaged_feats = feats.mean(dim=1)  # Final shape: [B, D]

            test_features.append(averaged_feats.cpu())
            test_filenames.extend(filenames)

            if i % 20 == 0:
                print(f"Test Extraction (TTA): Batch {i}/{len(test_loader)}")

    test_features = torch.cat(test_features, dim=0)

    torch.save({"features": test_features, "filenames": test_filenames}, SAVE_PATH_TEST)
    print(f">>> Saved test TTA features to {SAVE_PATH_TEST} | Shape: {test_features.shape}")


# %%
# ==========================================
#        MAIN EXECUTION
# ==========================================

def main():
    print(f"--- Loading {MODEL_TYPE} Backbone ---")
    model = model_utils.load_dino_model(
        model_type=MODEL_TYPE,
        weights_file=DINO_WEIGHTS,
        repo_dir=DINO_REPO
    )
    model.to(DEVICE)
    model.eval()

    # Execute extractions
    print("\n--- Extracting Train Features ---")
    extract_train_features(model)
    print("\n--- Extracting Test Features ---")
    extract_test_features_tta(model)

    print("\nDone!.")


if __name__ == "__main__":
    main()