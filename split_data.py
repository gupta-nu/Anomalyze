import os
import shutil
import random

# Get the absolute path of the current script's directory
base_dir = os.getcwd()  # This points to "NT data"
images_dir = os.path.join(base_dir, "raw_images")
masks_dir = os.path.join(base_dir, "path_to_masks")

output_dir = os.path.join(base_dir, "dataset")
splits = ["train", "val", "test"]
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

for split in splits:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# Debugging step: Check if the directories exist
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Images directory not found: {images_dir}")
if not os.path.exists(masks_dir):
    raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

images = sorted(os.listdir(images_dir))
masks = sorted(os.listdir(masks_dir))

image_mask_pairs = []
for img in images:
    base_name = os.path.splitext(img)[0]  # Get filename without extension
    possible_masks = [m for m in masks if base_name in m]  # Find corresponding mask
    if possible_masks:
        image_mask_pairs.append((img, possible_masks[0]))  # Store the first matching mask

random.shuffle(image_mask_pairs)

train_size = int(len(image_mask_pairs) * split_ratio["train"])
val_size = int(len(image_mask_pairs) * split_ratio["val"])

train_data = image_mask_pairs[:train_size]
val_data = image_mask_pairs[train_size:train_size + val_size]
test_data = image_mask_pairs[train_size + val_size:]

def copy_files(data, split):
    for img, mask in data:
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, split, "images", img))
        shutil.copy(os.path.join(masks_dir, mask), os.path.join(output_dir, split, "masks", mask))

copy_files(train_data, "train")
copy_files(val_data, "val")
copy_files(test_data, "test")

print("Data split completed successfully!")
