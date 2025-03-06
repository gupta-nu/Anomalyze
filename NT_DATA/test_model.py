import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained U-Net model
model = tf.keras.models.load_model("unet_final_model.h5")

# Load test data
dataset_dir = "NT_DATA/preprocessed"
X_test = np.load(os.path.join(dataset_dir, "test_images.npy"))
Y_test = np.load(os.path.join(dataset_dir, "test_masks.npy"))

# Add channel dimension to ground truth masks
Y_test = np.expand_dims(Y_test, axis=-1)

# Make predictions
Y_pred = model.predict(X_test)

# 🔍 Step 1: Check Raw Model Predictions Before Thresholding
plt.figure(figsize=(12, 4))
plt.imshow(Y_pred[0].squeeze(), cmap='viridis')  # 'viridis' improves visibility
plt.colorbar()
plt.title("Raw Model Prediction (Before Thresholding)")
plt.show()

# 🔍 Step 2: Check Training Labels
train_dataset_dir = "NT_DATA/preprocessed"
X_train = np.load(os.path.join(train_dataset_dir, "train_images.npy"))
Y_train = np.load(os.path.join(train_dataset_dir, "train_masks.npy"))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(X_train[0])
plt.title("Training Image")

plt.subplot(1, 2, 2)
plt.imshow(Y_train[0].squeeze(), cmap="gray")
plt.title("Training Mask")

plt.show()

# 🔍 Step 3: Adjust Threshold for Better Visibility
threshold = 0.2  # Lowering threshold to detect faint masks
Y_pred_thresh = (Y_pred > threshold).astype(np.uint8)

# Plot test results with adjusted threshold
for i in range(5):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i])
    plt.title("Ultrasound Image")

    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[i].squeeze(), cmap='gray')
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(Y_pred_thresh[i].squeeze(), cmap='gray')
    plt.title(f"Predicted Mask (Threshold {threshold})")

    plt.show()

# 🔍 Step 4: Define Dice Loss for Better Training Stability
from tensorflow.keras.losses import binary_crossentropy

def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))

def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
