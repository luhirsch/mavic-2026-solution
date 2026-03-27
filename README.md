# Team IDCOM Solution: 5th Multi-modal Aerial View Imagery Challenge: Classification (MAVIC-C)

### 3rd Place Overall | 1st Place for OOD Detection (Best AUROC & TNR@TPR95)

**Authors**: [Lucas Hirsch](https://luhirsch.github.io/), [Mike Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)

**Affiliation**: Institute for Imaging, Data and Communications (IDCOM), University of Edinburgh, UK

*View the official [CodaBench Results Leaderboard](https://www.codabench.org/competitions/12529/#/results-tab)*

## Solution Overview
This repository contains Team IDCOM's solution to the MAVIC-C 2026 competition held at CVPR as part of the PVBS workshop.

The solution proposes a cross-modal alignment framework that exploits the robust image recognition capabilities of the newly released DINOv3 foundation model.
We employ feature matching to align a trainable Synthetic Aperture Radar (SAR) feature space to a frozen Electro-Optical (EO) reference.

A core component of the architecture is the decoupling of the image classification and the Out of Distribution (OOD) detection.
A simple linear classifier handles classifiation, while OOD confidence scores are computed independently as the minimum Mahalanobis distance to class centroids in feature space.
This completely avoids the overconfidence flaws of standard logits or softmax probabilities.

![plot](./IDCOM_Architecture.png)



## Results
Team IDCOM achieved 3th place overall with a total score of 0.38.

**Performance metrics**:
 - Accuracy (Top-1): **24.8** %
 - AUROC: **0.78** (Competition Best)
 - TNR@TPR95: **0.27** (Competition Best)

----

## Installation

Major requirements:
 - PyTorch: DINOv3 requires a recent PyTorch version, this project uses PyTorch 2.6.
 - DINOv3 weights: DINOv3 weights can be downloaded from the [official repo](https://github.com/facebookresearch/dinov3). 
 - LoRA Finetuning: `peft` is a [parameter efficient fine tuning](https://github.com/huggingface/peft) library used for the LoRA fine tuning of the DINOv3 backbone.


Requirements can be installed with `pip`:

```pip install -r requirements.txt```


For a quick rundown of how to use the DINOv3 model, please check out the file [DINOv3_quickstart.ipynb](DINOv3_quickstart.ipynb).


## Configuration

All paths are configured in a single `config.yaml` file in the project root. To get started:
```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` with your local paths. The two entries that always require updating are the DINOv3 repo and weights paths, as these live outside the project directory. Dataset paths can be left as defaults if you place the MAVIC-C data in the `data/` folder (see [Data Setup](#data-setup) below).
```yaml
paths:
  dino_repo: "/path/to/dinov3"          # ← must update
  dino_weights: "/path/to/weights.pth"  # ← must update
  train_sar: "./data/train/SAR_Train"   # default, change if needed
  ...
```

> **Note:** Paths can be absolute or relative. Relative paths are resolved from the
> project root (where `config.yaml` lives), not from the `src/` directory.
> `config.yaml` is listed in `.gitignore` and will not be committed to the repository.

----

## Data Setup

Place the MAVIC-C 2025 dataset in the `data/` folder **OR** update the paths in `config.yaml` accordingly. The expected structure is:
````
data/
├── train/
│   ├── SAR_Train/
│   └── EO_Train/
├── val/                        # raw validation images (as provided by organizers)
├── val_organized/              # organized validation images (see below)
├── validation_reference.csv    # provided by organizers
└── test/
````

### Organizing the Validation Data
 
The validation images are provided as a flat folder of `.png` files alongside a
`validation_reference.csv` file that maps each image to its class and indicates
whether it is in-distribution (ID) or out-of-distribution (OOD).
 
The training and inference scripts expect the validation data to be organized into
subfolders per class, as required by PyTorch's `ImageFolder`. A helper script is
provided to do this automatically. Run it from the `data/` folder:
 
```bash
cd data
python ../organize_mavic_val_data_into_folders.py
```
 
This will create a `val_organized/` folder with the following structure:
 
```
val_organized/
├── IID/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── OOD/
```
 
Make sure the `val_organized` path in your `config.yaml` points to this folder.
 
---


## Usage

Below is an overview of how to use the solution. If you need any help running the code, please feel free to reach out.

0. For help using DINOv3, please check out the file [DINOv3_quickstart.ipynb](DINOv3_quickstart.ipynb) or refer to the [official repo](https://github.com/facebookresearch/dinov3)

1. **Training the Model**
To train the SAR backbone with LoRA and MMD alignment using the EO reference data, run:

2. Inference and OOD Detection:


