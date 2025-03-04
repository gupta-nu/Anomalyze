import os
import numpy as np
import cv2
from tqdm import tqdm

# Paths
dataset_dir = os.path.join(os.getcwd(), "dataset")
output_dir = "NT_DATA/preprocessed"


# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Set target image size (modify based on your model)
IMG_SIZE = (256, 256)

# Function to process images and masks
def process_data(split):
    images_path = os.path.join(dataset_dir, split, "images")
    masks_path = os.path.join(dataset_dir, split, "masks")

    # List images and masks
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    print(f"🔹 {split.upper()} - Total images: {len(image_files)}")
    print(f"🔹 {split.upper()} - Total masks: {len(mask_files)}")

    images = []
    masks = []

    for img_name in tqdm(image_files, desc=f"Processing {split} data"):
        img_path = os.path.join(images_path, img_name)

        # Convert ".jpg" filename to ".png" for mask
        base_name = os.path.splitext(img_name)[0]  # Remove extension
        mask_name = f"{base_name}.png"  # Ensure correct mask filename
        mask_path = os.path.join(masks_path, mask_name)

        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"⚠️ WARNING: No mask found for {img_name}")
            continue

        # Load image & mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure files are loaded correctly
        if image is None:
            print(f"❌ ERROR: Could not load image {img_path}")
            continue
        if mask is None:
            print(f"❌ ERROR: Could not load mask {mask_path}")
            continue

        # Resize and normalize
        image = cv2.resize(image, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE) / 255.0

        images.append(image)
        masks.append(mask)

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    print(f"✅ {split.upper()} processed: {images.shape}, {masks.shape}")

    # Save preprocessed data
    np.save(os.path.join(output_dir, f"{split}_images.npy"), images)
    np.save(os.path.join(output_dir, f"{split}_masks.npy"), masks)

# Process all dataset splits
for split in ["train", "val", "test"]:
    process_data(split)

print("✅ Preprocessing Complete! Data saved in:", output_dir)
