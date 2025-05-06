import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
import time
import argparse
import logging
from sklearn.metrics import jaccard_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate segmentation model and generate visualizations.')
    parser.add_argument('--data_dir', type=str, default="NT_DATA/preprocessed",
                        help='Directory containing preprocessed test data')
    parser.add_argument('--model_path', type=str, default="unet_best_model.h5",
                        help='Path to trained model file')
    parser.add_argument('--output_dir', type=str, default="evaluation_results",
                        help='Directory to save evaluation results')
    parser.add_argument('--num_visualizations', type=int, default=5,
                        help='Number of sample visualizations to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for model prediction')
    parser.add_argument('--small_component_threshold', type=int, default=100,
                        help='Minimum size for components in postprocessing')
    return parser.parse_args()

def load_test_data(data_dir):
    """Load test images and masks."""
    try:
        X_test = np.load(os.path.join(data_dir, "test_images.npy"))
        Y_test = np.load(os.path.join(data_dir, "test_masks.npy"))
        
        # Ensure masks are binary (0 or 1)
        Y_test = (Y_test > 0.5).astype(np.uint8)
        
        # Add channel dimension if needed
        if len(Y_test.shape) == 3:
            Y_test = np.expand_dims(Y_test, axis=-1)
            
        # Validation checks
        assert Y_test.max() <= 1 and Y_test.min() >= 0, "Masks must be binary (0-1)"
        assert Y_test.dtype == np.uint8, "Masks should be uint8 type"
        
        logger.info(f"✅ Loaded test data: {X_test.shape}, {Y_test.shape}")
        return X_test, Y_test
    except Exception as e:
        logger.error(f"❌ Error loading test data: {str(e)}")
        raise

def load_model(model_path):
    """Load the trained model."""
    try:
        # Custom objects dictionary for any custom layers/metrics
        custom_objects = {}
        
        # Try loading with tf.keras first
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except:
            # Fallback to pure Keras if tf.keras fails
            from keras.models import load_model
            model = load_model(model_path, custom_objects=custom_objects)
            
        logger.info(f"✅ Model loaded from: {model_path}")
        
        # Print model summary to log
        model.summary(print_fn=lambda x: logger.info(x))
        
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

def predict_with_model(model, X_test, batch_size=16):
    """Generate predictions with the model."""
    try:
        start_time = time.time()
        
        # Make predictions in batches to handle large datasets
        raw_preds = model.predict(X_test, batch_size=batch_size, verbose=1)
        
        # Validate model outputs
        assert raw_preds.max() <= 1.0 and raw_preds.min() >= 0.0, \
            f"Model outputs out of [0,1] range: {raw_preds.min()}-{raw_preds.max()}"
            
        elapsed = time.time() - start_time
        logger.info(f"✅ Predictions completed in {elapsed:.2f} seconds")
        
        return raw_preds
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise

def find_optimal_threshold(Y_test, raw_preds, metric='f1'):
    """Find optimal threshold for binary segmentation."""
    try:
        # Flatten arrays for threshold calculation
        y_true_flat = Y_test.flatten()
        y_pred_flat = raw_preds.flatten()
        
        precisions, recalls, thresholds = precision_recall_curve(y_true_flat, y_pred_flat)
        
        if metric == 'f1':
            # F1 score = 2 * (precision * recall) / (precision + recall)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'iou':
            # Calculate IoU at different thresholds (computationally intensive)
            ious = []
            for threshold in thresholds:
                y_pred_binary = (y_pred_flat > threshold).astype(np.uint8)
                iou = jaccard_score(y_true_flat, y_pred_binary, average='binary', zero_division=0)
                ious.append(iou)
            optimal_idx = np.argmax(ious)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"🏆 Optimal threshold ({metric}): {optimal_threshold:.4f}")
        
        return optimal_threshold
    except Exception as e:
        logger.error(f"❌ Error finding optimal threshold: {str(e)}")
        # Default fallback threshold
        return 0.5

def apply_postprocessing(binary_preds, min_component_size=100):
    """Apply morphological operations and component filtering."""
    try:
        # Create structural element for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        postprocessed_preds = []

        for mask in binary_preds:
            mask_squeezed = mask.squeeze()
            
            # Morphological closing (fill small holes)
            processed = cv2.morphologyEx(mask_squeezed, cv2.MORPH_CLOSE, kernel)
            
            # Optional: opening (remove small objects)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            # Connected components analysis
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed)
            
            if n_labels > 1:  # Has foreground components
                sizes = stats[1:, -1]  # Skip background component (index 0)
                processed = np.zeros_like(processed)
                
                # Keep only components above size threshold
                for i in range(n_labels-1):
                    if sizes[i] >= min_component_size:
                        processed[labels == i+1] = 1
                        
            postprocessed_preds.append(processed)

        # Add channel dimension back
        postprocessed_preds = np.expand_dims(np.array(postprocessed_preds), axis=-1)
        logger.info(f"✅ Postprocessing completed on {len(postprocessed_preds)} predictions")
        
        return postprocessed_preds
    except Exception as e:
        logger.error(f"❌ Error during postprocessing: {str(e)}")
        # Return original predictions if postprocessing fails
        return binary_preds

def dice_score(y_true, y_pred):
    """Calculate Dice coefficient."""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def calculate_metrics(y_true, y_pred, raw_preds=None):
    """Calculate multiple evaluation metrics."""
    try:
        # Flatten arrays for metric calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Basic metrics
        dice = dice_score(y_true_flat, y_pred_flat)
        iou = jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
        f1 = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
        sensitivity = tp / (tp + fn + 1e-6)  # Recall
        specificity = tn / (tn + fp + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        
        metrics = {
            'dice': dice,
            'iou': iou,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
        
        # ROC AUC if raw predictions are provided
        if raw_preds is not None:
            fpr, tpr, _ = roc_curve(y_true_flat, raw_preds.flatten())
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        
        return metrics
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {str(e)}")
        raise

def plot_curves(metrics, output_dir):
    """Plot and save ROC and Precision-Recall curves."""
    try:
        if 'fpr' not in metrics or 'tpr' not in metrics:
            logger.warning("⚠️ Cannot plot ROC curve: FPR or TPR not available")
            return
            
        # Prepare for Precision-Recall curve
        if 'y_true' in metrics and 'raw_preds' in metrics:
            precision_curve, recall_curve, _ = precision_recall_curve(
                metrics['y_true'].flatten(), 
                metrics['raw_preds'].flatten()
            )
        else:
            precision_curve = None
            recall_curve = None
            
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                 label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Precision-Recall curve
        if precision_curve is not None and recall_curve is not None:
            plt.subplot(1, 2, 2)
            plt.plot(recall_curve, precision_curve, color='blue', lw=2, label='Precision-Recall')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
        
        plt.tight_layout()
        
        # Save figure
        curves_path = os.path.join(output_dir, 'performance_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Performance curves saved to: {curves_path}")
    except Exception as e:
        logger.error(f"❌ Error plotting curves: {str(e)}")

def visualize_predictions(X, y_true, raw_preds, preds, postprocessed_preds, indices, output_dir):
    """Create and save visualizations of model predictions."""
    try:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        for i, idx in enumerate(indices):
            fig = plt.figure(figsize=(18, 4))
            
            # Prepare data for visualization
            image = X[idx]
            truth = y_true[idx]
            raw = raw_preds[idx]
            pred = preds[idx]
            post = postprocessed_preds[idx]
            
            # Calculate overlay (post-processed prediction on original image)
            overlay = create_overlay(image, post)
            
            images = [
                ("Original Image", image, None),
                ("Ground Truth", truth, 'gray'),
                ("Raw Output", raw, 'gray'),
                ("Thresholded", pred, 'gray'),
                ("Postprocessed", post, 'gray'),
                ("Overlay", overlay, None)
            ]
            
            for i, (title, data, cmap) in enumerate(images, 1):
                ax = fig.add_subplot(1, 6, i)
                ax.imshow(data.squeeze(), cmap=cmap)
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            vis_path = os.path.join(output_dir, 'visualizations', f'sample_{idx}.png')
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"✅ Visualization saved for sample {idx}")
        
        # Create a comparison figure (first sample only)
        create_comparison_figure(X[0], y_true[0], postprocessed_preds[0], output_dir)
    except Exception as e:
        logger.error(f"❌ Error creating visualizations: {str(e)}")

def create_overlay(image, mask):
    """Create an overlay of segmentation mask on the original image."""
    # Convert image to RGB if grayscale
    if image.shape[-1] == 1:
        image_rgb = np.repeat(image, 3, axis=-1)
    else:
        image_rgb = image
        
    # Create copy of image for overlay
    overlay = image_rgb.copy()
    
    # Create mask with red channel for overlay
    mask_rgb = np.zeros((*mask.shape[:-1], 3))
    mask_rgb[..., 0] = mask.squeeze()  # Red channel
    
    # Apply overlay with transparency
    alpha = 0.3
    overlay = (1-alpha) * image_rgb + alpha * mask_rgb
    
    # Normalize to [0, 1]
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def create_comparison_figure(image, truth, prediction, output_dir):
    """Create a side-by-side comparison figure."""
    try:
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image.squeeze())
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(132)
        plt.imshow(truth.squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Model prediction
        plt.subplot(133)
        plt.imshow(prediction.squeeze(), cmap='gray')
        plt.title('Model Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        comp_path = os.path.join(output_dir, 'comparison.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Comparison figure saved to: {comp_path}")
    except Exception as e:
        logger.error(f"❌ Error creating comparison figure: {str(e)}")

def save_results_summary(metrics_original, metrics_processed, args, output_dir):
    """Save evaluation results to a text file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write(f"Model Evaluation Summary - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"- Model: {args.model_path}\n")
            f.write(f"- Data directory: {args.data_dir}\n")
            f.write(f"- Component size threshold: {args.small_component_threshold}\n\n")
            
            # Metrics for original predictions
            f.write("Original Predictions Metrics:\n")
            for metric, value in metrics_original.items():
                if metric not in ['fpr', 'tpr', 'y_true', 'raw_preds']:
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n")
            
            # Metrics for post-processed predictions
            f.write("Post-processed Predictions Metrics:\n")
            for metric, value in metrics_processed.items():
                if metric not in ['fpr', 'tpr', 'y_true', 'raw_preds']:
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.4f}\n")
            f.write("\n")
            
            # Improvement percentage
            if 'dice' in metrics_original and 'dice' in metrics_processed:
                dice_improvement = ((metrics_processed['dice'] - metrics_original['dice']) / 
                                  metrics_original['dice']) * 100
                f.write(f"Dice Score Improvement: {dice_improvement:.2f}%\n")
                
            if 'iou' in metrics_original and 'iou' in metrics_processed:
                iou_improvement = ((metrics_processed['iou'] - metrics_original['iou']) / 
                                 metrics_original['iou']) * 100
                f.write(f"IoU Improvement: {iou_improvement:.2f}%\n")
                
        logger.info(f"✅ Evaluation summary saved to: {os.path.join(output_dir, 'evaluation_summary.txt')}")
    except Exception as e:
        logger.error(f"❌ Error saving results summary: {str(e)}")

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting model evaluation: {args.model_path}")
    
    try:
        # Load test data
        X_test, Y_test = load_test_data(args.data_dir)
        
        # Load model
        model = load_model(args.model_path)
        
        # Generate predictions
        raw_preds = predict_with_model(model, X_test, batch_size=args.batch_size)
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(Y_test, raw_preds)
        
        # Apply threshold
        preds = (raw_preds > optimal_threshold).astype(np.uint8)
        
        # Apply postprocessing
        postprocessed_preds = apply_postprocessing(
            preds, 
            min_component_size=args.small_component_threshold
        )
        
        # Calculate metrics for original predictions
        metrics_original = calculate_metrics(Y_test, preds, raw_preds)
        metrics_original['y_true'] = Y_test
        metrics_original['raw_preds'] = raw_preds
        
        # Calculate metrics for postprocessed predictions
        metrics_processed = calculate_metrics(Y_test, postprocessed_preds, raw_preds)
        metrics_processed['y_true'] = Y_test
        metrics_processed['raw_preds'] = raw_preds
        
        # Print metrics summary
        logger.info("\n📊 Performance Metrics:")
        logger.info(f"Dice Score - Original: {metrics_original['dice']:.4f} | Postprocessed: {metrics_processed['dice']:.4f}")
        logger.info(f"IoU - Original: {metrics_original['iou']:.4f} | Postprocessed: {metrics_processed['iou']:.4f}")
        logger.info(f"Sensitivity - Original: {metrics_original['sensitivity']:.4f} | Postprocessed: {metrics_processed['sensitivity']:.4f}")
        logger.info(f"Specificity - Original: {metrics_original['specificity']:.4f} | Postprocessed: {metrics_processed['specificity']:.4f}")
        
        # Plot curves
        plot_curves(metrics_processed, args.output_dir)
        
        # Generate visualizations
        sample_indices = np.random.choice(len(X_test), args.num_visualizations, replace=False)
        visualize_predictions(
            X_test, 
            Y_test,
            raw_preds,
            preds,
            postprocessed_preds,
            indices=sample_indices,
            output_dir=args.output_dir
        )
        
        # Save results summary
        save_results_summary(metrics_original, metrics_processed, args, args.output_dir)
        
        logger.info(f"✅ Evaluation completed successfully. Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()