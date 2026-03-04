"""
Script: train_eo_alignment.py
Purpose: Fine-tunes a SAR DINOv3 (ViT-S+) backbone using LoRA adapters,
         applying Maximum Mean Discrepancy (MMD) to align trainable SAR
         features with a frozen EO reference model. Handles class imbalance
         using a dynamically oversampled WeightedRandomSampler.
"""

import os
import sys
import platform
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
import numpy as np

# Choose path based on system
windowsPath = r"M:\projects"
linuxPath = "/home/s2807393/RDS/projects"
PROJECT_FOLDER = windowsPath if platform.system() == "Windows" else linuxPath
sys.path.append(PROJECT_FOLDER)

# Custom utility functions
import myutils
import dino_utils
import unicornv2_utils

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
MODEL_TYPE = "vits_plus"
IMAGE_SIZE = 128
CONFIG = {
    "BATCH_SIZE": 256,
    "EPOCHS": 10,
    "LR": 0.8e-5,
    "CHECKPOINT_NAME": f"dinov3_{MODEL_TYPE}_lora_finetune_CELoss_EOAlign_{IMAGE_SIZE}_v13.pth",
    "LORA_R": 20,
    "LORA_ALPHA": 40,
    "LORA_TARGET_MODULES": ["qkv"],
    "LORA_DROPOUT": 0.05,
    "LAMBDA_ALIGN": 0.45,
}

BASE_DIR = os.path.join(PROJECT_FOLDER, "dino_tests")
DATA_DIR = os.path.join(PROJECT_FOLDER, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "train", "SAR_Train")
TRAIN_DIR_EO = os.path.join(DATA_DIR, "MAVIC_C_2025", "train", "EO_Train")
VAL_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "val_organized", "IID")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "unicornv2_classification", "lora_checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = myutils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

unique_log_name = CONFIG['CHECKPOINT_NAME']
logger = myutils.setup_script_logger(__name__, filename=(unique_log_name[:-4] + ".log"))
logger.info(f"Using {DEVICE} device")

CLASS_NAMES = [
    "SUV", "box_truck", "bus", "flatbed_truck", "motorcycle",
    "pickup_truck", "pickup_truck_w_semi", "sedan", "semi_w_trailer", "van"
]

logger.info(f"""
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


# ==========================================
# 2. ARCHITECTURE & DATASET CLASSES
# ==========================================
class CrossModalDINOv3Classifier(nn.Module):
    def __init__(self, sar_backbone, eo_backbone, num_classes):
        super().__init__()
        self.sar_backbone = sar_backbone
        self.eo_backbone = eo_backbone

        for param in self.eo_backbone.parameters():
            param.requires_grad = False

        feat_dim = sar_backbone.embed_dim
        logger.info(f"Detected Embedding Dimension: {feat_dim}")
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, sar_x, eo_x=None):
        sar_features = F.normalize(self.sar_backbone(sar_x), dim=1)
        logits = self.head(sar_features)

        if eo_x is not None:
            with torch.no_grad():
                eo_features = F.normalize(self.eo_backbone(eo_x), dim=1)
            return logits, sar_features, eo_features
        return logits, sar_features, None


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.register_buffer('bandwidth_multipliers', mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bw = (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]
        return torch.exp(-L2_distances[None, ...] / bw).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class PairedSAREODataset(Dataset):
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


# ==========================================
# 3. TRAINING LOGIC
# ==========================================
def run_training():
    logger.info("--- 1. Loading Backbones ---")
    sar_backbone, _ = dino_utils.load_dino_model(MODEL_TYPE)
    eo_backbone, _ = dino_utils.load_dino_model(MODEL_TYPE)

    sar_backbone.to(DEVICE)
    eo_backbone.to(DEVICE)

    for param in sar_backbone.parameters(): param.requires_grad = False
    for param in eo_backbone.parameters(): param.requires_grad = False

    model = CrossModalDINOv3Classifier(sar_backbone, eo_backbone, num_classes=len(CLASS_NAMES))

    logger.info("--- 2. Injecting LoRA ---")
    peft_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        target_modules=CONFIG["LORA_TARGET_MODULES"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        exclude_modules=["eo_backbone"],
        bias="none",
        modules_to_save=["head"]
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,d} || All params: {all_param:,d} || Trainable%: {100 * trainable_params / all_param:.4f}")
    model.to(DEVICE)

    logger.info("--- 3. Preparing Data ---")
    transform = dino_utils.make_dino_transform(model_type=MODEL_TYPE, resize_size=IMAGE_SIZE)

    train_sar_ds = unicornv2_utils.CachedImageFolder(TRAIN_DIR, cache_name="train_mavic_cache.pt", transform=transform)
    train_eo_ds = unicornv2_utils.CachedImageFolder(TRAIN_DIR_EO, cache_name="train_mavic_eo_cache.pt",
                                                    transform=transform)
    train_paired_ds = PairedSAREODataset(train_sar_ds, train_eo_ds)
    val_ds = unicornv2_utils.CachedImageFolder(VAL_DIR, cache_name="val_id_mavic_cache.pt", transform=transform)

    # Export class mapping for inference script alignment
    mapping_path = os.path.join(OUTPUT_DIR, "class_to_idx.json")
    with open(mapping_path, "w") as f:
        json.dump(train_sar_ds.class_to_idx, f)
    logger.info(f"Saved class_to_idx mapping to {mapping_path}")

    logger.info("Calculating sampler weights...")
    targets = train_sar_ds.targets
    class_counts = np.bincount(targets)
    weights = 1. / (class_counts + 1e-5)
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(train_paired_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)

    criterion_classification = nn.CrossEntropyLoss().to(DEVICE)
    criterion_alignment = MMDLoss().to(DEVICE) if CONFIG["LAMBDA_ALIGN"] > 0 else nn.Identity().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

    logger.info("--- 4. Starting Training ---")
    best_f1 = 0.0

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0

        for i, (sar_images, eo_images, labels) in enumerate(train_loader):
            sar_images, eo_images, labels = sar_images.to(DEVICE), eo_images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            logits, sar_feats, eo_feats = model(sar_images, eo_images)

            loss_classification = criterion_classification(logits, labels)
            loss_align = criterion_alignment(sar_feats, eo_feats) if CONFIG["LAMBDA_ALIGN"] > 0 else 0.0

            loss = ((1 - CONFIG["LAMBDA_ALIGN"]) * loss_classification) + (CONFIG["LAMBDA_ALIGN"] * loss_align)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 50 == 0:
                logger.info(
                    f"Epoch {epoch + 1} [Batch {i}/{len(train_loader)}]: Loss CLS: {loss_classification.item():.3f} | Loss Align: {loss_align if isinstance(loss_align, float) else loss_align.item():.3f} | Total: {loss.item():.3f}")

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(images)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        pred_counts = np.bincount(all_preds, minlength=len(CLASS_NAMES))
        pred_probs = pred_counts / len(all_preds)
        entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-8))
        balance_score = entropy / np.log(len(CLASS_NAMES))

        logger.info(
            f"Epoch {epoch + 1}/{CONFIG['EPOCHS']} | Avg Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} | Pred Balance: {balance_score:.4f}")

        dist_str = " | ".join([f"{CLASS_NAMES[i]}: {count}" for i, count in enumerate(pred_counts)])
        logger.info(f"   ↳ Pred Distribution -> {dist_str}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(OUTPUT_DIR, CONFIG["CHECKPOINT_NAME"])
            torch.save(model.state_dict(), save_path)
            logger.info(f">>> New Best Model Saved to {save_path}")


if __name__ == "__main__":
    run_training()