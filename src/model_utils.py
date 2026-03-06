"""
Funtions for working with the DINOv3 models

load_dino_model: For loading the DINOv3 model
make_dino_transform: For creating the DINOv3 transform
CrossModalDINOv3Classifier: For creating the DINOv3 model wrapper that integrates the SAR and EO backbones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
from torchvision.transforms import v2 #needed for dino transform


# %% Function to load a dino model
def load_dino_model(model_type, weights_file, repo_dir):

    print(f"Attempting to load {model_type} from {weights_file}")

    # Quick check to ensure files exists
    if not os.path.exists(weights_file):
        print(f"Warning: Weights file not found at {weights_file}")
        sys.exit(-1)

    if not os.path.exists(weights_file):
        print(f"Warning: DINOv3 not found at {repo_dir}")
        sys.exit(-1)

    model = torch.hub.load(repo_dir,
                           model=model_type,
                           source='local',
                           weights=weights_file,
                           trust_repo=True)  # Local
    print(f"{model_type} loaded succesfully")

    return model


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

# %% Function to make the required DINOv3 transform (check the official GitHub repo for more details)

def make_dino_transform(resize_size: int = 256):
    """
    Transform needed to pass images through DINOv3. The values of the mean and std are the ImageNet mean and std.
    Use: transform = make_dino_transform(resize_size=n) where n can be any number
    :param model_type: type of model (satellite or object)
    :param resize_size: output image size
    """
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    return v2.Compose([to_tensor, resize, to_float, normalize])

# %% Function to get 5 views of an image and apply the DINOv3 transform
# The view of the image are (normal, horizontal flip, zoomed, 20% brighter, 20% darker)
def make_tta_transforms(resize_size=128):
    to_image = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # 1. Standard View
    t_base = v2.Compose([
        to_image,
        v2.Resize((resize_size, resize_size), antialias=True),
        to_float,
        normalize
    ])

    # 2. Flipped View
    t_flip = v2.Compose([
        to_image,
        v2.RandomHorizontalFlip(p=1.0),  # Force flip
        v2.Resize((resize_size, resize_size), antialias=True),
        to_float,
        normalize
    ])

    # 3. Zoomed View
    zoom_size = int(resize_size * 1.15)
    t_zoom = v2.Compose([
        to_image,
        v2.Resize((zoom_size, zoom_size), antialias=True),
        v2.CenterCrop((resize_size, resize_size)),
        to_float,
        normalize
    ])

    # 4. Brighter View
    t_bright = v2.Compose([
        to_image,
        v2.ColorJitter(brightness=(1.2, 1.2)),
        v2.Resize((resize_size, resize_size), antialias=True),
        to_float,
        normalize
    ])

    # 5. Darker View
    t_dark = v2.Compose([
        to_image,
        v2.ColorJitter(brightness=(0.8, 0.8)),
        v2.Resize((resize_size, resize_size), antialias=True),
        to_float,
        normalize
    ])

    return [t_base, t_flip, t_zoom, t_bright, t_dark]