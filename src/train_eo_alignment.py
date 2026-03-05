
"""
Script: train_eo_alignment.py
Purpose: Fine tunes a SAR DINOv3 (ViT-S+) backbone using LoRA adapters.
         Applies Maximum Mean Discrepancy (MMD) to align the trainable SAR
         features with a frozen EO reference model. Handles severe class
         imbalance using a dynamically oversampled WeightedRandomSampler.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
import numpy as np
import os, sys, platform



# %%
MODEL_TYPE = "vits_plus"  # ViT-Small
IMAGE_SIZE = 16 * 8  # PATCHSIZE * nPatches
CONFIG = {
    "BATCH_SIZE": 256,
    "EPOCHS": 10,
    "LR": 0.8e-5,
    "CHECKPOINT_NAME": f"dinov3_{MODEL_TYPE}_lora_finetune_"
                       f"CELoss_EOAlign_{IMAGE_SIZE}_v13.pth",

    # LoRA Configuration
    "LORA_R": 28,
    "LORA_ALPHA": 2 * 56,  # twice the rank
    "LORA_TARGET_MODULES": ["qkv"],  # 'qkv' matches the layer names you found: blocks.0.attn.qkv
    "LORA_DROPOUT": 0.05,

    # Loss configuration
    "LAMBDA_ALIGN": 0.45, # Weight for the alignment loss (MMD). 0 to disable.
}

BASE_DIR = os.path.join(PROJECT_FOLDER, "dino_tests")
DATA_DIR = os.path.join(PROJECT_FOLDER, "data")


# ==========================================
# 1. SETTINGS
# ==========================================
TRAIN_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "train", "SAR_Train")
TRAIN_DIR_EO = os.path.join(DATA_DIR, "MAVIC_C_2025", "train", "EO_Train")
VAL_DIR = os.path.join(DATA_DIR, "MAVIC_C_2025", "val_organized", "IID")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "unicornv2_classification", "lora_checkpoints")


if torch.cuda.is_available():
    DEVICE = myutils.get_freer_gpu()
else:
    DEVICE = "cpu"

unique_log_name = CONFIG['CHECKPOINT_NAME']
logger = myutils.setup_script_logger(__name__, filename=(unique_log_name[:-4] + ".log"))  # to remove the .pth
logger.info(f"Using {DEVICE} device")

CLASS_NAMES = [
    "SUV", "box_truck", "bus", "flatbed_truck", "motorcycle",
    "pickup_truck", "pickup_truck_w_semi", "sedan", "semi_w_trailer", "van"
]

# %%

# --- Log Configuration ---
logger.info(f"""
==========================================
    RUNNING EXPERIMENT: {MODEL_TYPE}
    CHECKPOINT:         {CONFIG['CHECKPOINT_NAME']}
==========================================
    Batch Size:   {CONFIG["BATCH_SIZE"]}
    Epochs:       {CONFIG["EPOCHS"]}
    Learning Rate: {CONFIG["LR"]}
    Image Size:   {IMAGE_SIZE}


    # LoRA Config:
    LoRA Rank (r): {CONFIG["LORA_R"]}
    LoRA Alpha:    {CONFIG["LORA_ALPHA"]}
    LoRA Target Modules: {CONFIG["LORA_TARGET_MODULES"]}
    LoRA Dropout:   {CONFIG["LORA_DROPOUT"]}

    # Loss Config:
    Focal Gamma: {CONFIG["FOCAL_GAMMA"]}
    Classification Loss Weight: {1 - CONFIG["LAMBDA_ALIGN"]}
    Alignment Loss Weight: {CONFIG["LAMBDA_ALIGN"]}
==========================================
""")


# %%

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
        logger.info(f"Detected Embedding Dimension: {feat_dim}")

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



# %% MMD Loss
class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # Registering as a buffer ensures it moves to the correct device automatically
        self.register_buffer(
            'bandwidth_multipliers',
            mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2

        # Calculate bandwidth and reshape for broadcasting
        bw = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers
        bw = bw[:, None, None]

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

# %% Custom Dataset for paired images (SAR and EO)
class PairedSAREODataset(Dataset):
    """
    Wraps two CachedImageFolders (SAR and EO) and yields them together.
    Assumes strictly identical file structures and alphabetical file naming
    between the two directories so that indices perfectly align.
    """

    def __init__(self, sar_dataset, eo_dataset):
        assert len(sar_dataset) == len(eo_dataset), "SAR and EO datasets must have the exact same number of samples!"
        self.sar_dataset = sar_dataset
        self.eo_dataset = eo_dataset

    def __len__(self):
        return len(self.sar_dataset)

    def __getitem__(self, idx):
        sar_img, sar_target = self.sar_dataset[idx]
        eo_img, eo_target = self.eo_dataset[idx]

        # Failsafe to ensure the directories haven't misaligned
        if sar_target != eo_target:
            raise ValueError(
                f"Target mismatch at index {idx}! SAR label: {sar_target}, EO label: {eo_target}. Check your directory alignment.")

        return sar_img, eo_img, sar_target

# %%


# ==========================================
# 3. TRAINING LOGIC
# ==========================================
def run_training():
    logger.info("--- 1. Loading Backbone ---")
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
    model = CrossModalDINOv3Classifier(sar_backbone, eo_backbone, num_classes=len(CLASS_NAMES))

    # --- 2. APPLY LoRA ---
    logger.info("--- 2. Injecting LoRA ---")
    peft_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        target_modules=CONFIG["LORA_TARGET_MODULES"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        exclude_modules=["eo_backbone"],  # CRITICAL: Shields the EO backbone from LoRA
        bias="none",
        modules_to_save=["head"],
        # IMPORTANT: Save 'head' AND 'projector' if using projector
        # modules_to_save=["head", "projector"],
    )

    model = get_peft_model(model, peft_config)
    # Capture PEFT's trainable parameter output into your logger
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    model.to(DEVICE)

    # ------- 3. DATA SETUP --------
    logger.info("--- 3. Preparing Data ---")
    # Standard DINO Transform
    transform = dino_utils.make_dino_transform(model_type=MODEL_TYPE, resize_size=IMAGE_SIZE)

    # Load SAR and EO train data
    train_sar_ds = unicornv2_utils.CachedImageFolder(TRAIN_DIR, cache_name="train_mavic_cache.pt", transform=transform)
    train_eo_ds = unicornv2_utils.CachedImageFolder(TRAIN_DIR_EO, cache_name="train_mavic_eo_cache.pt", transform=transform)
    train_paired_ds = PairedSAREODataset(train_sar_ds, train_eo_ds)     # and Wrap them in the paired dataset for the MMD dual-stream



    # Validation dataset (SAR only)
    val_ds = unicornv2_utils.CachedImageFolder(VAL_DIR, cache_name="val_id_mavic_cache.pt", transform=transform)


    # ----- LOSS SETUP -----
    # Weighted Sampler logic to handle Imbalance
    logger.info("Calculating sampler weights...")
    targets = train_sar_ds.targets
    class_counts = np.bincount(targets)
    weights = 1. / (class_counts + 1e-5)  # Avoid div/0
    samples_weights = weights[targets]
    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

    # If USING sampler, we set shuffle=False in DataLoader and pass the sampler
    train_loader = DataLoader(train_paired_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler, num_workers=4)

    # If NOT using sampler, for use with weighted loss instead. Set shuffle=True to ensure randomness.
    # train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], num_workers=4, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=4)

    # -------------- LOSSES SETUP ------------
    # ----- WEIGHTED CE LOSS SETUP -----
    # # Loss Weights for CrossEntropy
    # loss_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    # # Normalize weights so they sum roughly to num_classes
    # loss_weights = loss_weights / loss_weights.sum() * len(CLASS_NAMES)
    # criterion_ce = nn.CrossEntropyLoss(weight=loss_weights)

    # --- Unweighted CE Loss (if using sampler, we can use unweighted loss) ---
    # criterion_ce = nn.CrossEntropyLoss()   # Unweighted version

    # --- SUPERVISED Focal Loss + CE Loss (or FOCAL LOSS) ---
    if CONFIG["FOCAL_GAMMA"] > 0:
        criterion_classification = FocalLoss(alpha=None, gamma=CONFIG["FOCAL_GAMMA"]).to(DEVICE)  # Focal Loss
    else:
        criterion_classification = nn.CrossEntropyLoss().to(DEVICE)  # Cross Entropy Loss

    # --- ALIGNMENT LOSS ---
    if CONFIG["LAMBDA_ALIGN"] > 0:
        criterion_alignment = MMDLoss().to(DEVICE)
    else:
        criterion_alignment = nn.Identity().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

    # --- 4. TRAINING LOOP ---
    logger.info("--- 4. Starting Training ---")
    best_f1 = 0.0

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0

        for i, (sar_images, eo_images, labels) in enumerate(train_loader):
            sar_images = sar_images.to(DEVICE)
            eo_images = eo_images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward Pass returns TWO things now
            logits, sar_feats, eo_feats = model(sar_images, eo_images)

            # 1. Classification Loss (Trains the Head + Guides Backbone)
            loss_classification = criterion_classification(logits, labels)

            # 2. Alignemnt Loss (Structures Backbone)
            if CONFIG["LAMBDA_ALIGN"] > 0:
                loss_align = criterion_alignment(sar_feats, eo_feats)
            else:
                loss_align = 0.0

            # Total Loss (Weighted Sum)
            loss =  ((1-CONFIG["LAMBDA_ALIGN"]) * loss_classification) + (CONFIG["LAMBDA_ALIGN"] * loss_align)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % (50) == 0:
                logger.info(
                    f"Epoch {epoch + 1} [Batch {i}/{len(train_loader)}]: Loss CLS: {loss_classification.item():.3f} | Loss Align: {loss_align.item():.3f} | Total: {loss.item():.3f}")

        scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(images)  # We ignore features during val = model(images)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        # Prediction balance tracking
        pred_counts = np.bincount(all_preds,
                                  minlength=len(CLASS_NAMES))  # 1. Count how many times each class was predicted
        pred_probs = pred_counts / len(all_preds)  # 2. Convert to probabilities (proportions of the total batch)
        entropy = -np.sum(
            pred_probs * np.log(pred_probs + 1e-8))  # 3. Calculate Shannon Entropy (Add 1e-8 to avoid log(0) errors)
        max_entropy = np.log(
            len(CLASS_NAMES))  # 4. Normalize it (0.0 = Collapsed to one class, 1.0 = Perfectly balanced)
        balance_score = entropy / max_entropy

        logger.info(
            f"Epoch {epoch + 1}/{CONFIG["EPOCHS"]} | Avg Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} | Pred Balance: {balance_score:.4f}")

        # Log the actual raw distribution so you can visually verify the spread
        dist_str = " | ".join([f"{CLASS_NAMES[i]}: {count}" for i, count in enumerate(pred_counts)])
        logger.info(f"   ↳ Pred Distribution -> {dist_str}")

        # Save Best
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(OUTPUT_DIR, CONFIG["CHECKPOINT_NAME"])
            torch.save(model.state_dict(), save_path)
            logger.info(f">>> New Best Model Saved to {save_path}")


if __name__ == "__main__":
    run_training()