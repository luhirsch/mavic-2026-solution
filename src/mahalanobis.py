"""
Functions for fitting the Mahalanobis distance model
and processing data with it
"""

import numpy as np
from sklearn.covariance import LedoitWolf

def fit_mahalanobis_lw_model(X_train, y_train):
    """
    Fits a Mahalanobis Distance model using the Ledoit-Wolf shrinkage estimator.

    This calculates:
    1. The centroid (mean vector) for each class.
    2. A shared covariance matrix (inverse/precision) across all classes.

    Args:
        X_train (np.array): Training features of shape [N_samples, N_features].
        y_train (np.array): Training labels of shape [N_samples].

    Returns:
        class_means (np.array): Centroids [N_classes, N_features].
        precision_matrix (np.array): Inverse Covariance [N_features, N_features].
    """
    # Determine number of classes
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]
    print(f"Fitting Mahalanobis for {num_classes} classes with {num_features} features...")

    # Calculate Class Means (Centroids)
    class_means = np.zeros((num_classes, num_features))
    for c in range(num_classes):
        mask = (y_train == c)
        class_means[c] = X_train[mask].mean(axis=0)

    # Center the Data (Subtract class mean from samples)
    X_centered = np.zeros_like(X_train)
    for c in range(num_classes):
        mask = (y_train == c)
        X_centered[mask] = X_train[mask] - class_means[c]

    # 3. Fit Ledoit-Wolf Estimator to get Precision Matrix
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
            means, precision = fit_mahalanobis_lw_model(X_train, y_train)

            # Run the function
            scores, predictions = get_mahalanobis_scores(X_test, means, precision)

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
            #  Calculate the difference vector (x - u)
            diff = sample - class_means[c]

            #  Apply the Precision Matrix (Inverse Covariance)
            # distance = (x-u).T * Precision * (x-u)
            d = np.dot(np.dot(diff, precision_matrix), diff)
            dists.append(d)

        # Find the Winner and Save Results
        # Save results: Score is negative distance (closer to 0 is better)
        # High Score = Good (In-Distribution)
        # Low Score = Bad (Out-of-Distribution / Unknown)
        scores.append(-np.min(dists))
        preds.append(np.argmin(dists))

    return np.array(scores), np.array(preds)

