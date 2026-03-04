import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.covariance import LedoitWolf
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import MinCovDet
import pandas as pd
import re
import os, sys
from imblearn.under_sampling import NearMiss, ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
import torch
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader


# %% ------ Function for saving a Submission to the MAVIC-C Challenge
def save_submission(pred_idxs, scores, filenames_csv_path, output_path):
    """
    Generates and saves the competition submission CSV.

    Args:
        pred_idxs (list or np.array): The predicted class indices in Loader format (0=SUV).
        scores (list or np.array): The OOD confidence scores (e.g., negative Mahalanobis distance).
        filenames_csv_path (str): Path to the 'dino_unicornv2_test_features_file_names.csv' file.
        output_path (str): Full path where the result CSV should be saved.
    """
    # 1. Define Mapping (Loader Index -> Official ID)
    PRED_TO_OFFICIAL_MAP = {
        0: 1, 1: 4, 2: 7, 3: 6, 4: 5,
        5: 2, 6: 8, 7: 0, 8: 9, 9: 3
    }

    # 2. Load and Clean Filenames
    df_files = pd.read_csv(filenames_csv_path)
    image_filenames = df_files['image_id'].tolist()
    clean_ids = [re.sub(r'\D', '', str(fname)) for fname in image_filenames]

    # 3. Map Predictions
    official_class_ids = [PRED_TO_OFFICIAL_MAP[idx] for idx in pred_idxs]

    # 4. Create DataFrame
    submission_df = pd.DataFrame({
        'image_id': clean_ids,
        'class_id': official_class_ids,
        'score': scores
    })

    # 5. Format and Save
    submission_df['score'] = submission_df['score'].round(4)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(submission_df.head(5))




# %% ----- Classification models -------
def fit_mahalanobis_lw_model(X_train, y_train, cov_algo='ledoit-wolf'):
    """
    Fits a Mahalanobis Distance model using the Ledoit-Wolf shrinkage estimator.

    This calculates:
    1. The centroid (mean vector) for each class.
    2. A shared covariance matrix (inverse/precision) across all classes.

    Args:
        X_train (np.array): Training features of shape [N_samples, N_features].
        y_train (np.array): Training labels of shape [N_samples].
        cov_algo (str): The algorithm used to compute the covariance matrix.
                        'ledoit-wolf' or 'empirical' or 'minCovDet'
    Returns:
        class_means (np.array): Centroids [N_classes, N_features].
        precision_matrix (np.array): Inverse Covariance [N_features, N_features].
    """
    # 0. Determine  number of classes
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    print(f"Fitting Mahalanobis for {num_classes} classes with {num_features} features...")

    # 1. Calculate Class Means (Centroids)
    class_means = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        mask = (y_train == c)
        class_means[c] = X_train[mask].mean(axis=0)

    # 2. Center the Data (Subtract class mean from samples)
    X_centered = np.zeros_like(X_train)
    for c in range(num_classes):
        mask = (y_train == c)
        X_centered[mask] = X_train[mask] - class_means[c]

    # 3. Fit Ledoit-Wolf Estimator to get Precision Matrix
    # It handles cases where we have few samples (like the Semi w/ Trailer) much better
    # than standard EmpiricalCovariance, preventing the matrix from becoming singular.

    if cov_algo == 'empirical':
        print("Estimating Covariance Matrix (Empirical Covariance)...")
        cov_est = EmpiricalCovariance(assume_centered=True)
        cov_est.fit(X_centered)
    elif cov_algo == 'minCovDet':
        print("Estimating Covariance Matrix (Minimum Covariance Determinant)...")
        cov_est = MinCovDet(assume_centered=True, support_fraction=0.75)
        cov_est.fit(X_centered)
    else: # 'leidot-wolf'
        print("Estimating Covariance Matrix (Ledoit-Wolf)...")
        cov_est = LedoitWolf(assume_centered=True)
        cov_est.fit(X_centered)

    print("Mahalanobis Model Fitted Successfully.")
    return class_means, cov_est.precision_


# Function to get the
def get_mahalanobis_scores(X_test, class_means, precision_matrix):
    """
        Calculates Mahalanobis distance for test samples against class centroids.

        Args:
            X_test (np.ndarray): [n_samples, n_features]
            class_means (np.ndarray): [n_classes, n_features]
            precision_matrix (np.ndarray): Inverse covariance [n_features, n_features]

        Returns:
            scores (np.ndarray): Negative minimum distance (Higher = More confident).
            preds (np.ndarray): Predicted class indices.

        Usage:
            means, precision = unicornv2_utils.fit_mahalanobis_lw_model(X_train, y_train)

            # Run the function
            scores, predictions = unicornv2_utils.get_mahalanobis_scores(X_test, means, precision)

            # Check results
            print(f"Predicted Class: {predictions[0]}")
            print(f"Confidence Score: {scores[0]}") # e.g., -15.4 (Closer to 0 is better)
    """

    scores = []
    preds = []
    n_classes = len(class_means)

    # Loop through every image in the test set
    for i in range(len(X_test)):
        sample = X_test[i]
        dists = []

        # Compare this image against every Class Centroid
        for c in range(n_classes):
            # 1. Calculate the difference vector (x - u)
            diff = sample - class_means[c]

            # 2. Apply the Precision Matrix (Inverse Covariance)
            # distance = (x-u).T * Precision * (x-u)
            # This "weighs" the difference based on feature variance
            d = np.dot(np.dot(diff, precision_matrix), diff)
            dists.append(d)

        # 3. Find the Winner and Save Results
        # Save results: Score is negative distance (closer to 0 is better)
        # High Score = Good (In-Distribution)
        # Low Score = Bad (Out-of-Distribution / Unknown)
        scores.append(-np.min(dists))
        preds.append(np.argmin(dists))

    return np.array(scores), np.array(preds)

# %% Utility functions for calculating metrics related to the MAVIC_C Challenge
def calc_tnr_at_tpr95(y_true, y_scores):
    """
    Calculates the True Negative Rate (OOD Rejection Rate) when
    the True Positive Rate (ID Retention Rate) is 95%.

    Args:
        y_true: Array of 1 (ID) and 0 (OOD).
        y_scores: Array of confidence scores (Higher = More likely ID).
    """
    # 1. Sort scores in descending order (High confidence first)
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # 2. Find total ID samples (Positives)
    total_positives = np.sum(y_true_sorted == 1)

    # 3. Find the threshold index where we capture 95% of ID samples
    cum_positives = np.cumsum(y_true_sorted)
    tpr_threshold_idx = np.argmax(cum_positives >= 0.95 * total_positives)

    # 4. The score at this index is our "95% Recall Threshold"
    threshold = y_scores_sorted[tpr_threshold_idx]

    # 5. Calculate TNR: Fraction of OOD samples that fall BELOW this threshold
    # We look at the original unsorted arrays for clarity
    ood_scores = y_scores[y_true == 0]
    total_ood = len(ood_scores)

    if total_ood == 0:
        return 0.0

    rejected_ood = np.sum(ood_scores < threshold)

    return rejected_ood / total_ood


def get_validation_metrics(y_pred_id, y_scores_id, y_scores_ood, y_true_id_labels=None):
    """
    Calculates Accuracy, AUCROC, and TNR@TPR95.

    ID = In Distribution
    OOD = Out of Distribution

    Args:
        y_pred_id:      Predicted class IDs for the ID Validation set (e.g., [0, 4, ...])
        y_scores_id:    Confidence scores for the ID Validation set.
        y_scores_ood:   Confidence scores for the OOD Validation set.
        y_true_id_labels: (Optional) Ground truth labels for IID set to calc Accuracy.
                          If None, accuracy is skipped.

    Returns:
        Dictionary containing 'acc', 'auc', 'tnr95'.

    Usage:
        Assuming you have your features loaded as numpy arrays:
        X_iid, y_iid (Labels)
        X_ood

        1. Get Predictions and Scores from your Classifier
        (Example using your Mahalanobis/LogReg logic)
            preds_iid = ... (Your class predictions for IID)
            scores_iid = ... (Your negative distances/max logits for IID)
            scores_ood = ... (Your negative distances/max logits for OOD)
        # 2. Call the function
            metrics = get_validation_metrics(
            y_pred_id=preds_iid,
            y_scores_id=scores_iid,
            y_scores_ood=scores_ood,
            y_true_id_labels=y_iid
            )

        # 3. Access individual values if needed
        print(f"My final score check: {metrics['auc']}")

    """

    results = {}

    # --- 1. Accuracy (VAL_IID) ---
    if y_true_id_labels is not None:
        acc = accuracy_score(y_true_id_labels, y_pred_id)
        results['acc'] = acc * 100  # percentage
        print(f"Accuracy (IID): {results['acc']:.2f}%")

    # --- 2. Prepare OOD Metric Arrays ---
    # ID samples = Label 1, OOD samples = Label 0
    y_true_binary = np.concatenate([
        np.ones(len(y_scores_id)),
        np.zeros(len(y_scores_ood))
    ])

    y_scores_combined = np.concatenate([
        y_scores_id,
        y_scores_ood
    ])

    # --- 3. AUCROC ---
    auc = roc_auc_score(y_true_binary, y_scores_combined)
    results['auc'] = auc
    print(f"AUC ROC:        {auc:.4f}")

    # --- 4. TNR @ TPR95 ---
    tnr95 = calc_tnr_at_tpr95(y_true_binary, y_scores_combined)
    results['tnr95'] = tnr95
    print(f"TNR@TPR95:      {tnr95:.4f}")

    return results


# %% For undersampling majority classes
def undersample_majority_classes(X, y, strategy="random", max_samples=5000, random_seed = 42):
    """
    Smarter undersampling using imblearn, tied to a specific random seed.

    STRATEGIES:
    - "random":   Randomly deletes majority samples. Fastest method. Changing the seed gives
                  completely different data, making it perfect for Seed Ensembles.
    - "nearmiss": Keeps only the majority samples that are physically closest to the minority
                  classes. Deletes the "easy" safe samples to force boundary learning.
    - "cluster":  Runs K-Means clustering to find representative "centroids" of the majority
                  class. Preserves the overall shape of the data but is very slow.

    Caps the number of samples per class at `max_samples`.
    If a class has fewer than `max_samples` (like Class 9), it keeps all of them.
    """
    print(f"Applying {strategy.upper()} undersampling (Max: {max_samples}, Seed: {random_seed})...")

    # imblearn requires numpy arrays
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    # 1. Cap majority classes at 'max_samples'
    class_counts = np.bincount(y_np)
    sampling_strategy = {}
    for class_id, count in enumerate(class_counts):
        if count > max_samples:
            sampling_strategy[class_id] = max_samples
        else:
            sampling_strategy[class_id] = count

            # 2. Choose the algorithm and inject the seed
    if strategy == "nearmiss":
        # NearMiss is deterministic (exact distance-based). Seed has no effect here.
        sampler = NearMiss(version=1, sampling_strategy=sampling_strategy, n_jobs=-1)
    elif strategy == "cluster":
        # Seed changes the K-Means initialization, yielding slightly different centroids
        sampler = ClusterCentroids(sampling_strategy=sampling_strategy, random_state=random_seed)
    else:
        # Seed completely changes which majority samples are kept (Best for ensembling!)
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_seed)

    # 3. Apply the undersampling
    X_resampled, y_resampled = sampler.fit_resample(X_np, y_np)

    print(f"Original dataset shape: {X_np.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")

    # Convert back to PyTorch tensors
    return torch.from_numpy(X_resampled).float(), torch.from_numpy(y_resampled).long()



class CachedImageFolder(VisionDataset):
    # Check unicornv2_gemini_dino_LORA_FT.py for how to use this
    """
    A drop-in replacement for datasets.ImageFolder that caches the
    directory scan results to a file, reducing initialization time
    from minutes to milliseconds on subsequent runs.
    """
    def __init__(self, root, cache_name="folder_cache.pt", transform=None):
        super().__init__(root, transform=transform)
        self.cache_path = os.path.join(root, cache_name)
        self.loader = default_loader

        # 1. Check if we already scanned this folder
        if os.path.exists(self.cache_path):
            print(f"Loading dataset structure from cache: {self.cache_path}")
            data = torch.load(self.cache_path)
            self.classes = data['classes']
            self.class_to_idx = data['class_to_idx']
            self.samples = data['samples']
            self.targets = data['targets']
        else:
            # 2. If no cache, do the slow scan once using the standard ImageFolder
            print(f"Cache not found. Scanning directory {root} (this may take a while)...")
            temp_folder = ImageFolder(root)
            self.classes = temp_folder.classes
            self.class_to_idx = temp_folder.class_to_idx
            self.samples = temp_folder.samples
            self.targets = temp_folder.targets

            # 3. Save the results so we never have to scan it again
            torch.save({
                'classes': self.classes,
                'class_to_idx': self.class_to_idx,
                'samples': self.samples,
                'targets': self.targets
            }, self.cache_path)
            print(f"Successfully cached dataset structure to {self.cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # Load the image
        sample = self.loader(path)
        # Apply transforms if any
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target




class AddMultiplicativeSpeckleNoise(object):
    """Simulates SAR speckle (Multiplicative Gaussian Noise)"""
    def __init__(self, prob=0.5, variance=0.04):
        self.prob = prob
        self.variance = variance

    def __call__(self, tensor):
        if torch.rand(1).item() < self.prob:
            std = self.variance ** 0.5
            noise = torch.randn_like(tensor) * std
            noisy_tensor = tensor + (tensor * noise)
            return torch.clamp(noisy_tensor, 0.0, 1.0)
        return tensor