import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('/home/ananya/Downloads/college files/icmr-internship/NT-fetal-anomaly-detection/NT_DATA/unet_best_model.h5')  # Make sure the path is correct

# Print model architecture
print("="*50)
print("Model Summary:")
print("="*50)
model.summary()

# Optional: Save summary to text file
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))