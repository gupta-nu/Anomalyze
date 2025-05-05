import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess image and mask data for segmentation.')
    parser.add_argument('--dataset_dir', type=str, default=os.path.join(os.getcwd(), "dataset"),
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default="NT_DATA/preprocessed",
                        help='Output directory for preprocessed data')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Target image size as (width, height)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing to manage memory usage')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='Number of worker processes')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization samples')
    return parser.parse_args()

def process_image_mask_pair(data):
    """Process a single image-mask pair."""
    img_path, mask_path, img_size = data
    
    # Load image & mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure files are loaded correctly
    if image is None:
        return None, None, f"Could not load image {img_path}"
    if mask is None:
        return None, None, f"Could not load mask {mask_path}"
    
    # Resize and normalize
    image = cv2.resize(image, img_size)
    mask = cv2.resize(mask, img_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    
    return image, mask, None

def create_visualization(images, masks, output_dir, split, num_samples=5):
    """Create and save visualization of sample images and their masks."""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select random samples or first few if less than num_samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display image
        axes[0].imshow(images[idx])
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Display mask
        axes[1].imshow(masks[idx], cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        plt.suptitle(f"{split.capitalize()} Sample {i+1}")
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(vis_dir, f"{split}_sample_{i+1}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
    
    logger.info(f"✅ Created {len(indices)} visualizations for {split} set in {vis_dir}")

def process_data_in_batches(split, dataset_dir, output_dir, img_size, batch_size, num_workers, visualize=False):
    """Process data in batches to manage memory usage."""
    images_path = os.path.join(dataset_dir, split, "images")
    masks_path = os.path.join(dataset_dir, split, "masks")
    
    # Ensure directories exist
    if not os.path.exists(images_path):
        logger.error(f"❌ Images directory not found: {images_path}")
        return
    if not os.path.exists(masks_path):
        logger.error(f"❌ Masks directory not found: {masks_path}")
        return
    
    # List images and masks
    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        logger.error(f"❌ No image files found in {images_path}")
        return
    
    logger.info(f"🔹 {split.upper()} - Total images found: {len(image_files)}")
    
    # Prepare data for processing
    process_data = []
    skipped = 0
    
    for img_name in image_files:
        img_path = os.path.join(images_path, img_name)
        
        # Convert image filename to mask filename (handling different extensions)
        base_name = os.path.splitext(img_name)[0]  # Remove extension
        mask_name = f"{base_name}.png"  # Try PNG extension for mask
        mask_path = os.path.join(masks_path, mask_name)
        
        # If not found, try other common extensions
        if not os.path.exists(mask_path):
            for ext in ['.jpg', '.jpeg']:
                alt_mask_name = f"{base_name}{ext}"
                alt_mask_path = os.path.join(masks_path, alt_mask_name)
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            logger.warning(f"⚠️ No mask found for {img_name}, skipping")
            skipped += 1
            continue
        
        process_data.append((img_path, mask_path, tuple(img_size)))
    
    if skipped > 0:
        logger.warning(f"⚠️ Skipped {skipped} images due to missing masks")
    
    # Process in batches
    all_images = []
    all_masks = []
    total_batches = (len(process_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(process_data))
        batch_data = process_data[start_idx:end_idx]
        
        batch_images = []
        batch_masks = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_image_mask_pair, data): data for data in batch_data}
            
            for future in tqdm(as_completed(futures), total=len(batch_data), 
                               desc=f"Processing {split} batch {batch_idx+1}/{total_batches}"):
                image, mask, error = future.result()
                if error:
                    logger.error(f"❌ {error}")
                else:
                    batch_images.append(image)
                    batch_masks.append(mask)
        
        if batch_images:
            all_images.extend(batch_images)
            all_masks.extend(batch_masks)
            
            # Free memory after each batch
            batch_images = None
            batch_masks = None
    
    if not all_images:
        logger.error(f"❌ No valid image-mask pairs processed for {split}")
        return
    
    # Convert lists to NumPy arrays
    all_images = np.array(all_images, dtype=np.float32)
    all_masks = np.array(all_masks, dtype=np.float32)
    
    logger.info(f"✅ {split.upper()} processed: {all_images.shape}, {all_masks.shape}")
    
    # Create visualizations if requested
    if visualize and len(all_images) > 0:
        create_visualization(all_images, all_masks, output_dir, split)
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, f"{split}_images.npy"), all_images)
    np.save(os.path.join(output_dir, f"{split}_masks.npy"), all_masks)
    
    # Calculate and log dataset statistics
    img_mean = all_images.mean(axis=(0, 1, 2))
    img_std = all_images.std(axis=(0, 1, 2))
    mask_mean = all_masks.mean()
    mask_coverage = np.mean(all_masks > 0.5)  # Percentage of positive mask pixels
    
    logger.info(f"📊 {split.upper()} Statistics:")
    logger.info(f"   - Image Mean (RGB): {img_mean}")
    logger.info(f"   - Image Std (RGB): {img_std}")
    logger.info(f"   - Mask Mean: {mask_mean:.4f}")
    logger.info(f"   - Mask Coverage: {mask_coverage:.2%}")
    
    return all_images.shape, all_masks.shape

def main():
    """Main function to run the preprocessing pipeline."""
    args = parse_arguments()
    
    # Print configuration
    logger.info("🚀 Starting preprocessing with configuration:")
    logger.info(f"   - Dataset directory: {args.dataset_dir}")
    logger.info(f"   - Output directory: {args.output_dir}")
    logger.info(f"   - Image size: {args.img_size}")
    logger.info(f"   - Batch size: {args.batch_size}")
    logger.info(f"   - Worker processes: {args.workers}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    with open(os.path.join(args.output_dir, "preprocessing_config.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Process all dataset splits
    splits = ["train", "val", "test"]
    results = {}
    
    for split in splits:
        split_dir = os.path.join(args.dataset_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"⚠️ Split directory not found: {split_dir}, skipping")
            continue
        
        result = process_data_in_batches(
            split=split,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.workers,
            visualize=args.visualize
        )
        
        if result:
            results[split] = result
    
    # Summary
    logger.info("✅ Preprocessing Complete! Data saved in: " + args.output_dir)
    logger.info("📊 Summary:")
    for split, (img_shape, mask_shape) in results.items():
        logger.info(f"   - {split.upper()}: {img_shape[0]} samples, Image shape: {img_shape[1:]} - Mask shape: {mask_shape[1:]}")

if __name__ == "__main__":
    main()