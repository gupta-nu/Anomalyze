import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import jaccard_score

# Load test images
dataset_dir = "NT_DATA/preprocessed"
X_test = np.load(os.path.join(dataset_dir, "test_images.npy"))
Y_test = np.load(os.path.join(dataset_dir, "test_masks.npy"))

# Reshape masks
Y_test = np.expand_dims(Y_test, axis=-1)

# Load trained model
model = tf.keras.models.load_model("unet_best_model.h5")

# Predict masks
preds = model.predict(X_test)
preds = (preds > 0.3).astype(np.uint8)  # Apply thresholding

# Apply dilation to predictions
kernel = np.ones((3, 3), np.uint8)
preds_dilated = []
for mask in preds:
    mask_squeezed = mask.squeeze()  # Remove channel dimension for cv2 processing
    dilated = cv2.dilate(mask_squeezed, kernel, iterations=1)
    preds_dilated.append(dilated)
preds_dilated = np.expand_dims(np.array(preds_dilated), axis=-1)  # Add channel dimension back

# Convert ground truth and predictions to binary masks
Y_test_bin = [mask.astype(np.uint8) for mask in Y_test]
preds_dilated_bin = [mask.astype(np.uint8) for mask in preds_dilated]  # Already binary after dilation

# Evaluate Dice and IoU
def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

dice_scores = [dice_score(Y_test_bin[i].squeeze(), preds_dilated_bin[i].squeeze()) for i in range(len(Y_test))]
iou_scores = [jaccard_score(Y_test_bin[i].flatten(), preds_dilated_bin[i].flatten(), average='binary') for i in range(len(Y_test))]

print(f"✅ Average Dice Score after dilation: {np.mean(dice_scores):.4f}")
print(f"✅ Average IoU Score after dilation: {np.mean(iou_scores):.4f}")

# Visualize Predictions
def visualize_predictions(X_test, Y_test, preds, preds_dilated, index=0):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(X_test[index], cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(Y_test[index].squeeze(), cmap='gray')
    plt.title("Ground Truth Mask")

    plt.subplot(1, 4, 3)
    plt.imshow(preds[index].squeeze(), cmap='gray')
    plt.title("Predicted Mask")

    plt.subplot(1, 4, 4)
    plt.imshow(preds_dilated[index].squeeze(), cmap='gray')
    plt.title("Dilated Predicted Mask")

    plt.show()

# Show a few samples
for i in range(5):
    visualize_predictions(X_test, Y_test, preds, preds_dilated, index=i)