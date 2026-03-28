"""
Script: inference.py
Purpose: Generates submission CSV for the MAVIC-C 2025 Competition.
        1. Obtain class_id from fine tuned model
        2. Train Mahalanobis OOD detector on train features
        3. Obtain OOD scores from TTA features
        4. Assemble into submission CSV
"""

import os
import re
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
import yaml
from pathlib import Path

import model_utils
import mahalanobis

# %%
# ==============================================
#  DIRECTORY SETTINGS (obtained from config.yaml)
# ==============================================

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Paths
DINO_REPO = Path(cfg["paths"]["dino_repo"])
DINO_WEIGHTS_VITS = Path(cfg["paths"]["dino_weights_vits_plus"])
TEST_DIR = Path(cfg["paths"]["test"])

# Outputs (submission CSV)
OUTPUT_DIR = "./output"
OUTPUT_CSV = os.path.join("submission.csv")

# Features
TRAIN_FEATURES_MAHA = os.path.join(OUTPUT_DIR, "train_features_vitl_sat.pt") # Train features for Mahalanbis
TEST_FEATURES_MAHA = os.path.join(OUTPUT_DIR, "test_features_TTA.pt") # Test TTA features for Mahalanbis


# ==========================================
# 1. SETTINGS
# ==========================================
MODEL_TYPE = "dinov3_vits16plus"
IMAGE_SIZE = 128
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA & Checkpoint config
LORA_CONFIG = {
    "LORA_R": 28,             # Updated to match your training script!
    "LORA_ALPHA": 56,         # Updated to match your training script!
    "LORA_TARGET_MODULES": ["qkv"],
    "LORA_DROPOUT": 0.05,
}

CHECKPOINT_NAME = f"{MODEL_TYPE}_LoRA_finetune_{IMAGE_SIZE}.pth"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)


# ---- Class Mappings ----
# Alphabetical order used during PyTorch ImageFolder Training
TRAIN_CLASSES = [
    "SUV", "box_truck", "bus", "flatbed_truck", "motorcycle",
    "pickup_truck", "pickup_truck_w_trailer", "sedan", "semi_w_trailer", "van"
]


# The official competition mapping
OFFICIAL_CLASS_MAP = {
    0: "sedan",
    1: "SUV",
    2: "pickup_truck",
    3: "van",
    4: "box_truck",
    5: "motorcycle",
    6: "flatbed_truck",
    7: "bus",
    8: "pickup_truck_w_trailer",
    9: "semi_w_trailer"
}

# Reverse the official map to get Name -> ID
NAME_TO_OFFICIAL_ID = {v: k for k, v in OFFICIAL_CLASS_MAP.items()}

# Create Map: PyTorch Training Index -> Official Competition ID
TRAIN_IDX_TO_OFFICIAL_ID = {
    i: NAME_TO_OFFICIAL_ID[name] for i, name in enumerate(TRAIN_CLASSES)
}



# %%
# ==========================================
# 2. DATASET DEFINITION
# ==========================================
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        # Read all files and SORT them alphabetically
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
# 3. INFERENCE LOGIC (STEP 1 - Obtain class_id)
# ==========================================
def get_class_predictions():
    """Passes test images to the fine-tuned model and returns class_id predictions."""
    print("--- Loading Backbone ---")
    sar_backbone = model_utils.load_dino_model(model_type=MODEL_TYPE, weights_file=DINO_WEIGHTS_VITS,
                                               repo_dir=DINO_REPO)
    eo_backbone = model_utils.load_dino_model(model_type=MODEL_TYPE, weights_file=DINO_WEIGHTS_VITS,
                                              repo_dir=DINO_REPO)

    sar_backbone.to(DEVICE)
    eo_backbone.to(DEVICE)

    for param in sar_backbone.parameters(): param.requires_grad = False
    for param in eo_backbone.parameters(): param.requires_grad = False

    # Wrap both backbones into dual stream architecture
    model = model_utils.CrossModalDINOv3Classifier(sar_backbone, eo_backbone, num_classes=len(TRAIN_CLASSES ))

    print(f"--- Injecting LoRA and Loading Weights ---")
    peft_config = LoraConfig(
        r=LORA_CONFIG["LORA_R"],
        lora_alpha=LORA_CONFIG["LORA_ALPHA"],
        target_modules=LORA_CONFIG["LORA_TARGET_MODULES"],
        lora_dropout=LORA_CONFIG["LORA_DROPOUT"],
        bias="none",
        modules_to_save=["head"],
    )

    model = get_peft_model(model, peft_config)

    # Load Weights from checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("--- Preparing Test Data ---")
    transform = model_utils.make_dino_transform(resize_size=IMAGE_SIZE)
    test_ds = FlatFolderDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    # ---  PREDICTION LOOP ---
    print(f"--- Running Inference on {len(test_ds)} images ---")

    # Using a dictionary to store results keyed by image_id for easy merging later
    predictions = {}

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

            for j in range(len(preds)):
                train_idx = preds[j]
                filename = fnames[j]

                # Extract Numeric Image ID from filename (e.g. "test_00123.png" -> 123)
                match = re.search(r'\d+', filename)
                image_id = int(match.group()) if match else -1

                # Map to Official ID
                official_id = TRAIN_IDX_TO_OFFICIAL_ID[train_idx]

                predictions[image_id] = {
                    'class_id': official_id,
                }

            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}")

    return predictions
#%%
# ==========================================
#           MAIN
# ==========================================

if __name__ == "__main__":

    # --- STEP 1: Pass test images to model and make class_id predictions ---
    print("\n--- Get class predictions (Top1 Acc) ---")
    preds_dict = get_class_predictions()

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(preds_dict, orient='index').reset_index()
    df.rename(columns={'index': 'image_id'}, inplace=True)
    df = df.sort_values('image_id')

    # --- STEP 2: Train Mahalanobis on train features ---
    print("\n--- Step 2: Training Mahalanobis Model ---")
    train_data = torch.load(TRAIN_FEATURES_MAHA)
    X_train = train_data["features"].numpy()
    y_train = train_data["labels"].numpy()

    class_means, precision_matrix = mahalanobis.fit_mahalanobis_lw_model(X_train, y_train)

    # --- STEP 3: Pass test TTA features to Mahalanobis and get OOD scores ---
    print("\n--- Step 3: Calculating Mahalanobis OOD Scores for Test Set (TTA) ---")
    test_data = torch.load(TEST_FEATURES_MAHA)
    X_test = test_data["features"].numpy()
    test_filenames = test_data["filenames"]

    # Get OOD Scores
    maha_scores, _ = mahalanobis.get_mahalanobis_scores(X_test, class_means, precision_matrix)

    # --- STEP 4: Combine class_id and OOD scores into submission CSV ---
    print("\n--- Step 4: Assembling Final Submission CSV ---")

    # Map the Mahalanobis scores to their corresponding image_ids
    maha_scores_dict = {}
    for fname, score in zip(test_filenames, maha_scores):
        match = re.search(r'\d+', fname)
        image_id = int(match.group()) if match else -1
        maha_scores_dict[image_id] = score

    # 'score' column (Mahalanobis distance)
    df['score'] = df['image_id'].map(maha_scores_dict)

    # Ensure correct column order: image_id, class_id, score
    df = df[['image_id', 'class_id', 'score']]

    # Save final CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Final submission saved to: {OUTPUT_CSV}")

    # Distribution Check
    official_id_counts = df['class_id'].value_counts().to_dict()
    table_data = []

    for official_id in range(10):
        vehicle_type = OFFICIAL_CLASS_MAP[official_id]
        count = official_id_counts.get(official_id, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        table_data.append({"Class #": official_id, "Type": vehicle_type, "Count": count, "%": f"{percentage:.1f}%"})

    print("\n--- Final Class Distribution ---")
    print(pd.DataFrame(table_data).to_markdown(index=False))


