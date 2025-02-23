import os
import json
import numpy as np
import cv2
from glob import glob 
from labelme import utils 

json_dir= "annotations"
output_dir= "path_to_masks"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for json_file in glob(os.path.join(json_dir, "*.json")):
    with open(json_file) as f:
        data= json.load(f)

    img_shape = (data["imageHeight"], data["imageWidth"])
    mask = np.zeros(img_shape, dtype=np.uint8)

    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=255)  # NT region = white (255)

    mask_filename = os.path.join(output_dir, os.path.basename(json_file).replace(".json", ".png"))
    cv2.imwrite(mask_filename, mask)

print(" Mask conversion done!")
