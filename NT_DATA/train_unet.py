import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

# Data augmentation
data_gen_args = dict(rotation_range=20,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     brightness_range=[0.8, 1.2],
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Create TensorFlow dataset to fix error
def create_dataset(X, Y, batch_size=4):
    def generator():
        for img, mask in zip(X, Y):
            yield img, mask
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
    ))
    return dataset.batch(batch_size).shuffle(100)

train_dataset = create_dataset(X_train, Y_train)
val_dataset = create_dataset(X_val, Y_val)

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

# Callbacks
checkpoint = ModelCheckpoint("unet_best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[checkpoint, early_stopping])


# Save final model
model.save("unet_final_model.h5")
print("✅ Model trained and saved as 'unet_final_model.h5'")

# Plot training history (accuracy and loss)
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
