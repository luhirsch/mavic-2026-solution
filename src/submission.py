import numpy as np
import pandas as pd
import re



# %% ------ Function for saving a Submission to the MAVIC-C Challenge
def save_submission(pred_idxs, scores, filenames_csv_path, output_path):
    """
    Handles the formatting and exporting of final test predictions.
    Maps the internal ImageFolder class indices to the official challenge
    submission format and saves the results as a CSV for CodaBench.

    Use: Generates and saves the competition submission CSV.

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
