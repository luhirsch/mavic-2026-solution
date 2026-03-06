"""
Funtions for working with the DINOv3 models

load_dino_model: For loading the DINOv3 model
make_dino_transform: For creating the DINOv3 transform
CrossModalDINOv3Classifier: For creating the DINOv3 model wrapper that integrates the SAR and EO backbones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



# %% Model wrapper.
# Takes the DINOv3 SAR backbone, EO backbone and adds a linear head.
# Freezes the EO backbone completely.

# ==========================================
#             MODEL WRAPPER
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

        # Classification head (only for SAR features)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, sar_x, eo_x=None):
        # Get SAR features
        sar_features = self.sar_backbone(sar_x)
        sar_features = F.normalize(sar_features, dim=1)

        # Predict using SAR features
        logits = self.head(sar_features)

        # If EO data is provided (training), get EO features
        if eo_x is not None:
            with torch.no_grad():
                eo_features = F.normalize(self.eo_backbone(eo_x), dim=1)
            return logits, sar_features, eo_features

        # If no EO data is provided (validation/testing), return None for EO
        return logits, sar_features, None