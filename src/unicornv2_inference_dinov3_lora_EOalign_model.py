"""
Script: inference_dinov3_lora.py
Purpose: Loads a fine-tuned DINOv3 LoRA checkpoint and generates predictions
         on the test set. Handles class mapping from Alphabetical (Training)
         to Official Competition IDs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from PIL import Image
import re


import os, sys, platform

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
import unicornv2_utils

# %%
BASE_DIR =  os.path.join(PROJECT_FOLDER, "dino_tests")
DATA_DIR = os.path.join(PROJECT_FOLDER, "data")


# ==========================================
# 1. SETTINGS
# ==========================================
MODEL_TYPE = "vits_plus"  # ViT-Small
IMAGE_SIZE = 128    # Matches patch size 16 (16 * 16 = 256)
LORA_CONFIG = { # I need to get this from the script that trained the model (by hand). Annoying, i know.
    "LORA_R": 64,
    "LORA_ALPHA": 64*2,  # twice the rank
    "LORA_TARGET_MODULES": ["qkv"],  # 'qkv' matches the layer names you found: blocks.0.attn.qkv
    "LORA_DROPOUT": 0.05,
}
# CHECKPOINT_NAME = f"dinov3_{MODEL_TYPE}_lora_finetune_SupConLoss_CELoss_Resampling_128_v2.pth"
CHECKPOINT_NAME = f"dinov3_vits_plus_lora_finetune_CELoss_EOAlign_128_v12.pth"
OUTPUT_CSV_NAME = f"submission_{CHECKPOINT_NAME[:-4]}.csv" # [:-4] to remove the .pth
BATCH_SIZE = 64


# -- Folders --
TRAIN_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "train", "SAR_Train")
VAL_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "val_organized", "IID")
TEST_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "test")
# TEST_DIR_TTA
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "unicornv2_classification", "lora_checkpoints")

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)

# OUTPUT
OUTPUT_CSV = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)

# CLASS MAPPINGS
# 1. The alphabetical order used during Training (ImageFolder default)
TRAIN_CLASSES = [
    "SUV", "box_truck", "bus", "flatbed_truck", "motorcycle",
    "pickup_truck", "pickup_truck_w_semi", "sedan", "semi_w_trailer", "van"
]

# 2. Map Name -> Official Competition ID
NAME_TO_OFFICIAL_ID = {
    "sedan": 0,
    "SUV": 1,
    "pickup_truck": 2,
    "van": 3,
    "box_truck": 4,
    "motorcycle": 5,
    "flatbed_truck": 6,
    "bus": 7,
    "pickup_truck_w_semi": 8,
    "semi_w_trailer": 9
}

# The official competition mapping
OFFICIAL_CLASS_MAP = {
    0: "sedan",
    1: "SUV",
    2: "pickup truck",
    3: "van",
    4: "box truck",
    5: "motorcycle",
    6: "flatbed truck",
    7: "bus",
    8: "pickup truck w/ trailer",
    9: "semi truck w/ trailer"
}

# 3. Create Map: Training Index -> Official ID
TRAIN_IDX_TO_OFFICIAL_ID = {
    i: NAME_TO_OFFICIAL_ID[name] for i, name in enumerate(TRAIN_CLASSES)
}

if torch.cuda.is_available():
    DEVICE = myutils.get_freer_gpu()
else:
    DEVICE = "cpu"
print(f"Using {DEVICE} device")



# ==========================================
# 2. MODEL WRAPPER
# ==========================================
class CrossModalDINOv3Classifier(nn.Module):
    def __init__(self, sar_backbone, eo_backbone, num_classes):
        super().__init__()
        self.sar_backbone = sar_backbone
        self.eo_backbone = eo_backbone

        # Freeze the EO backbone completely
        for param in self.eo_backbone.parameters():
            param.requires_grad = False

        feat_dim = sar_backbone.embed_dim
        print(f"Detected Embedding Dimension: {feat_dim}")

        # Classification head attached ONLY to the SAR features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, sar_x, eo_x=None):
        # 1. Get SAR features (requires_grad = True, driven by LoRA)
        sar_features = self.sar_backbone(sar_x)
        sar_features = F.normalize(sar_features, dim=1)

        # 2. Predict using SAR features
        logits = self.head(sar_features)

        # 3. If EO data is provided (Training), get EO features
        if eo_x is not None:
            with torch.no_grad():
                eo_features = self.eo_backbone(eo_x)
                eo_features = F.normalize(eo_features, dim=1)
            return logits, sar_features, eo_features

        # 4. If no EO data is provided (Validation/Testing), return None for EO
        return logits, sar_features, None



# %% Test Folder Dataset

# ==========================================
# 2. DATASET DEFINITION
# ==========================================
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        # Read all files and SORT them alphabetically (Matches your provided logic)
        self.imgs = sorted([f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __getitem__(self, idx):
        # Load image
        fname = self.imgs[idx]
        img_path = os.path.join(self.root, fname)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return image AND filename so we can track IDs
        return image, fname

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 4. INFERENCE LOGIC
# ==========================================
def run_inference():
    print("--- 1. Loading Backbone ---")
    # Load SAR backbone
    sar_backbone, _ = dino_utils.load_dino_model(MODEL_TYPE)
    sar_backbone.to(DEVICE)
    for param in sar_backbone.parameters():
        param.requires_grad = False

    # Load EO backbone
    eo_backbone, _ = dino_utils.load_dino_model(MODEL_TYPE)
    eo_backbone.to(DEVICE)
    for param in eo_backbone.parameters():
        param.requires_grad = False

    # Wrap both backbones into your dual-stream architecture
    model = CrossModalDINOv3Classifier(sar_backbone, eo_backbone, num_classes=len(TRAIN_CLASSES ))


    print(f"--- 2. Loading LoRA from {CHECKPOINT_PATH} ---")
    peft_config = LoraConfig(
        r=LORA_CONFIG["LORA_R"],
        lora_alpha=LORA_CONFIG["LORA_ALPHA"],
        target_modules=LORA_CONFIG["LORA_TARGET_MODULES"],
        lora_dropout=LORA_CONFIG["LORA_DROPOUT"],
        bias="none",
        modules_to_save=["head"],
    )

    model = get_peft_model(model, peft_config)

    # Load Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print(f"--- 3. Preparing Data (Size: {IMAGE_SIZE}) ---")
    transform = dino_utils.make_dino_transform(
        model_type=MODEL_TYPE,
        resize_size=IMAGE_SIZE
    )

    test_ds = FlatFolderDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Found {len(test_ds)} test images.")

    # --- 4. PREDICTION LOOP ---
    print("--- 4. Running Inference ---")

    all_image_ids = []
    all_class_ids = []
    all_scores = []

    with torch.no_grad():
        for i, (images, fnames) in enumerate(test_loader):
            images = images.to(DEVICE)

            # Forward
            logits, _, _ = model(images)
            probs = torch.softmax(logits, dim=1)

            # Get Predictions
            max_probs, preds = torch.max(probs, dim=1)

            # Move to CPU
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            for j in range(len(preds)):
                train_idx = preds[j]
                score = max_probs[j]
                filename = fnames[j]

                # Extract Numeric Image ID from filename (e.g. "test_00123.png" -> 123)
                match = re.search(r'\d+', filename)
                if match:
                    image_id = int(match.group())
                else:
                    print(f"Warning: Could not parse ID from {filename}")
                    image_id = -1

                # Map to Official ID
                official_id = TRAIN_IDX_TO_OFFICIAL_ID[train_idx]

                all_image_ids.append(image_id)
                all_class_ids.append(official_id)
                all_scores.append(score)

            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}")

    # --- 5. SAVE CSV ---
    df = pd.DataFrame({
        'image_id': all_image_ids,
        'class_id': all_class_ids,
        'score': all_scores
    })

    # Sort by image_id to ensure competition format consistency
    df = df.sort_values('image_id')

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"--- Submission saved to {OUTPUT_CSV} ---")

    # Distribution Check
    # Count how many times each Official Competition ID appears in the CSV
    official_id_counts = df['class_id'].value_counts().to_dict()
    table_data = []
    total_preds = len(df)

    # Loop from 0 to 9 to keep the table in the exact order of the official documentation
    for official_id in range(10):
        vehicle_type = OFFICIAL_CLASS_MAP[official_id]

        count = official_id_counts.get(official_id, 0)
        percentage = (count / total_preds) * 100 if total_preds > 0 else 0

        table_data.append({
            "Class #": official_id,
            "Vehicle Type": vehicle_type,
            "Count": count,
            "Percentage": f"{percentage:.1f}%"
        })

    results_df = pd.DataFrame(table_data)
    print("\n--- Final Competition Distribution ---")
    print("\n" + results_df.to_markdown(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run_inference()