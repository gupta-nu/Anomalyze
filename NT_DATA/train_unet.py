import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate
import matplotlib.pyplot as plt

# Set paths
dataset_dir = "NT_DATA/preprocessed"

# Load preprocessed data
X_train = np.load(os.path.join(dataset_dir, "train_images.npy"))
Y_train = np.load(os.path.join(dataset_dir, "train_masks.npy"))
X_val = np.load(os.path.join(dataset_dir, "val_images.npy"))
Y_val = np.load(os.path.join(dataset_dir, "val_masks.npy"))

# Reshape masks to have an additional channel
Y_train = np.expand_dims(Y_train, axis=-1)
Y_val = np.expand_dims(Y_val, axis=-1)

# Define U-Net model
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    upconv1 = Conv2D(128, 3, activation='relu', padding='same')(up1)

    up2 = UpSampling2D(size=(2, 2))(upconv1)
    upconv2 = Conv2D(64, 3, activation='relu', padding='same')(up2)

    outputs = Conv2D(1, 1, activation='sigmoid')(upconv2)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = unet_model()

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=4)

# Save the model
model.save("unet_model.h5")
print("✅ Model trained and saved as 'unet_model.h5'")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
