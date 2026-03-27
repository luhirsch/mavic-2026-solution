"""
Script: train_eo_alignment.py
Purpose: Fine-tunes a SAR DINOv3 (ViT-S+) backbone using LoRA adapters,
         applying Maximum Mean Discrepancy (MMD) to align trainable SAR
         features with a frozen EO reference model. Handles class imbalance
         using a dynamically oversampled WeightedRandomSampler.

Update: ** please update config.yaml **

Outputs:
    - Model Checkpoint: Saves the best model based on F1-score on the validation set

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
import numpy as np
from mmd_loss import MMDLoss
import model_utils
import yaml
from pathlib import Path



# %%
# ==============================================
#  DIRECTORY SETTINGS (obtained from config.yaml)
# ==============================================

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# DINOv3 weights - Please update with your paths
DINO_WEIGHTS = Path(cfg["paths"]["dino_weights_vits_plus"])
DINO_REPO = Path(cfg["paths"]["dino_repo"])


# Data - Put MAVIC-C 2025 data in the "data" folder or update with your paths
TRAIN_DATA_SAR = Path(cfg["paths"]["train_sar"])
TRAIN_DATA_EO = Path(cfg["paths"]["train_eo"])
VAL_DATA = Path(cfg["paths"]["val_organized"]) # This folder is obtained after running "organize_mavic_val_into_folders"

# Outputs (model checkpoint)
# The model will be saved in the "output" folder as a .pth file
OUTPUT_DIR = "./output"


# %%
# ==========================================
#        TRAINING SETTINGS
# ==========================================
MODEL_TYPE = "dinov3_vits16plus"
IMAGE_SIZE = 128 # Size to which images will be resized for DINOv3 (should be a multiple of 16)
CONFIG = {
    # Training Configuration
    "BATCH_SIZE": 256,
    "EPOCHS": 25,
    "LR": 0.8e-5,
    "CHECKPOINT_NAME": f"{MODEL_TYPE}_LoRA_finetune_{IMAGE_SIZE}.pth",

    # LoRA Configuration
    "LORA_R": 28,
    "LORA_ALPHA": 2*28,
    "LORA_TARGET_MODULES": ["qkv"],
    "LORA_DROPOUT": 0.05,

    # Loss configuration
    "LAMBDA_ALIGN": 0.45,
}
RANDOM_SEED = 15
SAVE_PATH = os.path.join(OUTPUT_DIR, CONFIG["CHECKPOINT_NAME"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%

CLASS_NAMES = [
    "SUV", "box_truck", "bus", "flatbed_truck", "motorcycle",
    "pickup_truck", "pickup_truck_w_trailer", "sedan", "semi_w_trailer", "van"
]

print(f"""
==========================================
    RUNNING EXPERIMENT: {MODEL_TYPE}
    CHECKPOINT:         {CONFIG['CHECKPOINT_NAME']}
==========================================
    Batch Size:   {CONFIG["BATCH_SIZE"]}
    Epochs:       {CONFIG["EPOCHS"]}
    Learning Rate: {CONFIG["LR"]}
    Image Size:   {IMAGE_SIZE}

    LoRA Rank (r): {CONFIG["LORA_R"]}
    Classification Weight: {1 - CONFIG["LAMBDA_ALIGN"]}
    Alignment Weight:      {CONFIG["LAMBDA_ALIGN"]}
==========================================
""")



# %% Dataset Wrapper
# Takes the SAR and EO datasets and pairs them together for training.

# ==========================================
#             DATASET WRAPPER
# ==========================================
class PairedSAREODataset(Dataset):
    """
    Wraps two ImageFolder datasets (SAR and EO) and yields them together.
    Assumes strictly identical file structures and alphabetical file naming
    between the two directories so that indices align.
    """
    def __init__(self, sar_dataset, eo_dataset):
        assert len(sar_dataset) == len(eo_dataset), "SAR and EO datasets must match in length!"
        self.sar_dataset = sar_dataset
        self.eo_dataset = eo_dataset

    def __len__(self):
        return len(self.sar_dataset)

    def __getitem__(self, idx):
        sar_img, sar_target = self.sar_dataset[idx]
        eo_img, eo_target = self.eo_dataset[idx]

        if sar_target != eo_target:
            raise ValueError(f"Target mismatch at idx {idx}! SAR: {sar_target}, EO: {eo_target}.")
        return sar_img, eo_img, sar_target




# %%
# ==========================================
#            TRAINING LOGIC
# ==========================================
def run_training():
    print("--- Loading Backbones ---")
    sar_backbone = model_utils.load_dino_model(model_type=MODEL_TYPE, weights_file=DINO_WEIGHTS, repo_dir=DINO_REPO)
    eo_backbone = model_utils.load_dino_model(model_type=MODEL_TYPE, weights_file=DINO_WEIGHTS, repo_dir=DINO_REPO)

    sar_backbone.to(DEVICE)
    eo_backbone.to(DEVICE)

    for param in sar_backbone.parameters(): param.requires_grad = False
    for param in eo_backbone.parameters(): param.requires_grad = False

    # Wrap both backbones into dual stream architecture
    model = model_utils.CrossModalDINOv3Classifier(sar_backbone, eo_backbone, num_classes=len(CLASS_NAMES))

    # Apply LoRA - We use the peft library
    print("--- Injecting LoRA ---")
    peft_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        target_modules=CONFIG["LORA_TARGET_MODULES"], #Inject into qkv modules of transformer blocks
        lora_dropout=CONFIG["LORA_DROPOUT"],
        exclude_modules=["eo_backbone"], # Shield the EO backbone from LoRA
        bias="none",
        modules_to_save=["head"]
    )

    model = get_peft_model(model, peft_config)

    # Print the amount of trainable parameters after LoRA injection
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"Trainable params: {trainable_params:,d} || All params (2 DINOv3 + Head): {all_param:,d} || Trainable%: {100 * trainable_params / all_param:.4f}")
    model.to(DEVICE)

    print("--- Preparing Data ---")
    # Make DINOv3 transform
    transform = model_utils.make_dino_transform(resize_size=IMAGE_SIZE)

    # --- Set up Pytorch datasets ---
    # Load SAR and EO train data
    train_sar_ds = datasets.ImageFolder(TRAIN_DATA_SAR, transform=transform)
    train_eo_ds = datasets.ImageFolder(TRAIN_DATA_EO, transform=transform)
    train_paired_ds = PairedSAREODataset(train_sar_ds, train_eo_ds)

    # Load validation data
    val_ds = datasets.ImageFolder(VAL_DATA, transform=transform)


    # --- Set up PyTorch DataLoaders and Sampler---
    # Get weights for the WeightedRandomSampler
    print("Calculating sampler weights...")
    targets = train_sar_ds.targets
    class_counts = np.bincount(targets)
    weights = 1. / (class_counts + 1e-5)
    samples_weights = weights[targets]

    # Set seed and create sampler
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    sampler = WeightedRandomSampler(samples_weights,
                                    num_samples=len(samples_weights),
                                    generator=generator,
                                    replacement=True)


    # Create DataLoaders
    train_loader = DataLoader(train_paired_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)

    # Loss functions and training setup
    criterion_classification = nn.CrossEntropyLoss().to(DEVICE)
    criterion_alignment = MMDLoss().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

    # --- Training Loop ---
    print("--- Starting Training ---")
    best_f1 = 0.0 # Checkpointing model based on best f1-score on validation set

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0

        for i, (sar_images, eo_images, labels) in enumerate(train_loader):
            sar_images, eo_images, labels = sar_images.to(DEVICE), eo_images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass through the model
            logits, sar_feats, eo_feats = model(sar_images, eo_images)

            # Compute losses
            loss_classification = criterion_classification(logits, labels)
            loss_align = criterion_alignment(sar_feats, eo_feats)
            loss = ((1 - CONFIG["LAMBDA_ALIGN"]) * loss_classification) + (CONFIG["LAMBDA_ALIGN"] * loss_align)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1} [Batch {i}/{len(train_loader)}]: Loss CLS: {loss_classification.item():.3f}"
                      f" | Loss Align: {loss_align.item():.3f} | Total: {loss.item():.3f}")

        scheduler.step()

        # Run model through VALIDATION data to monitor performance and save the BEST MODEL based on f1-score
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(images) # We ignore features during validation
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']} | Avg Loss: {total_loss / len(train_loader):.4f} |"
              f" Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f">>> New Best Model Saved to {SAVE_PATH}")


if __name__ == "__main__":
    run_training()