import os
import cv2
import numpy as np

# INPUT folder (original clean images)
input_folder = "data/LOLv1/Test/input"

# OUTPUT folders
output_base = "data/LOLv1/Degraded"

alphas = [0.3, 0.1, 0.05]  # darkness levels
gammas = [2, 3]            # gamma levels

# Create output directories
for a in alphas:
    os.makedirs(f"{output_base}/dark_alpha_{a}", exist_ok=True)

for g in gammas:
    os.makedirs(f"{output_base}/gamma_{g}", exist_ok=True)

# Process images
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = img.astype(np.float32) / 255.0

    # -------- Linear Darkness --------
    for a in alphas:
        dark = img * a
        dark = np.clip(dark * 255, 0, 255).astype(np.uint8)

        save_path = f"{output_base}/dark_alpha_{a}/{img_name}"
        cv2.imwrite(save_path, dark)

    # -------- Gamma Darkness --------
    for g in gammas:
        gamma_img = np.power(img, g)
        gamma_img = np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

        save_path = f"{output_base}/gamma_{g}/{img_name}"
        cv2.imwrite(save_path, gamma_img)

print("Degraded images generated successfully!")