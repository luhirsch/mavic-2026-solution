# Team IDCOM Solution: 5th Multi-modal Aerial View Imagery Challenge: Classification (MAVIC-C)

### 4th Place Overall | 1st Place for OOD Detection (Best AUROC & TNR@TPR95)

*View the official [CodaBench Results Leaderboard](https://www.codabench.org/competitions/12529/#/results-tab)*

## Solution Overview
This repository contains Team IDCOM's solution to the MAVIC-C 2026 competition held at CVPR as part of the PVBS workshop.

The solution proposes a cross-modal alignment framework that exploits the robust image recognition capabilities of the newly released DINOv3 foundation model.
We employ feature matching to align a trainable Synthetic Aperture Radar (SAR) feature space to a frozen Electro-Optical (EO) reference.

A core component of the architecture is the decoupling of the image classification and the Out of Distribution (OOD) detection.
A simple linear classifier handles classifiation, while OOD confidence scores are computed independently as the minimum Mahalanobis distance to class centroids in feature space.
This completely avoids the overconfidence flaws of standard logits or softmax probabilities.

**Authors**: Lucas Hirsch, Mike Davies


**Affiliation**: Institute for Imaging, Data and Communications (IDCOM), University of Edinburgh, UK

## Results
Team IDCOM achieved 4th place overall with a total score of 0.38.

**Performance metrics**:
 - Accuracy (Top-1): **24.8** %
 - AUROC: **0.78** (Competition Best)
 - TNR@TPR95: **0.27** (Competition Best)

----

## Installation

DINOv3 weights can be downloaded from the [official repo](https://github.com/facebookresearch/dinov3). DINOv3 requires a recent PyTorch version, this project uses PyTorch 2.6.
`peft` is a [parameter efficient fine tuning](https://github.com/huggingface/peft) library used for the LoRA fine tuning of the DINOv3 backbone.



Requirements can be installed with `pip`:

```pip install -r requirements.txt```


----

## Usage

1. Training the Model
To train the SAR backbone with LoRA and MMD alignment using the EO reference data, run:

2. Inference and OOD Detection:


