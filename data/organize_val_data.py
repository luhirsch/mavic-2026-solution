import os
import shutil
import pandas as pd
from tqdm import tqdm

"""
Script to organize the validation folder into SUBFOLDERS for each CLASS based on the reference
given in validation_reference.csv

Steps:
 1. Separates the images into ID and OOD folders.
 2. The ID images are organized into subfolders based on the class name.
"""

# --- CONFIGURATION ---
CSV_PATH = 'validation_reference.csv'
SOURCE_DIR = 'val'  # Folder containing the raw images
DEST_DIR = 'val_organized'  # New folder to be created
EXTENSION = '.png'  # Explicitly set to .png


def organize_images():
    # 1. Load the reference CSV
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found.")
        return

    df = pd.read_csv(CSV_PATH)

    # 2. Define mapping for OOD_flag (0 -> IID, 1 -> OOD)
    subset_map = {0: 'IID', 1: 'OOD'}

    print(f"Starting organization of {len(df)} images...")

    success_count = 0
    missing_count = 0

    # 3. Iterate through every row in the CSV
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row['image_id']).strip()
        class_name = str(row['class']).strip()
        ood_flag = int(row['OOD_flag'])

        # Determine subset folder (IID or OOD)
        subset_name = subset_map.get(ood_flag, 'Unknown')

        # Build the destination path: validation_organized/IID/box_truck/
        target_folder = os.path.join(DEST_DIR, subset_name, class_name)
        os.makedirs(target_folder, exist_ok=True)

        # FORCE .png extension
        filename = image_id + EXTENSION

        src_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(target_folder, filename)

        # 4. Copy the file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            success_count += 1
        else:
            missing_count += 1
            # print(f"Missing: {filename}") # Uncomment to debug

    print("\n--- Summary ---")
    print(f"Successfully organized: {success_count} images")
    print(f"Missing files: {missing_count}")
    print(f"Output directory: {os.path.abspath(DEST_DIR)}")


if __name__ == "__main__":
    organize_images()