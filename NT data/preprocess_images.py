import os
from PIL import Image

# Path to raw images and processed output
input_folder = 'raw_images'
output_folder = 'processed_images'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Resize dimensions
resize_width, resize_height = 224, 224

# Process each image
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((resize_width, resize_height))
        
        # Save processed image
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)
        print(f'Processed: {filename}')

print("Image preprocessing completed!")
