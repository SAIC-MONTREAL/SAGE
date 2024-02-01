"""Script to change the alpha of all images within a directory and create clone folder with paler
iamges"""
import os
import numpy as np
import cv2

# Get the path to the folder containing the images
ROOT = os.getenv("SMARTHOME_ROOT", default=None)

folder_path = os.path.join(ROOT, "assets", "icons")
new_folder_path = os.path.join(ROOT, "assets", "transparent_icons")
alpha = 0.2

# Get the list of images in the folder
images = os.listdir(folder_path)

# Iterate through each image
for image in images:
    print(image)

    # Read the image
    img = cv2.imread(os.path.join(folder_path, image))

    # Change the alpha of the image
    img = cv2.addWeighted(
        img, alpha, (255 - np.zeros_like(img)).astype("uint8"), 1 - alpha, 0
    )

    # Save the image
    cv2.imwrite(os.path.join(new_folder_path, image), img)

# Print a message confirming that the images have been processed
print("Images processed!")
