import cv2
import os
import numpy as np

# Folders
mask_dir = "kvasir/Kvasir-SEG-Dataset/seg_dataset/labels/tests_masks"  
label_dir = "kvasir/Kvasir-SEG-Dataset/seg_dataset/yolo_labels"       
os.makedirs(label_dir, exist_ok=True)

for mask_file in os.listdir(mask_dir):
    if not mask_file.endswith((".jpg", ".png")):
        continue

    mask_path = os.path.join(mask_dir, mask_file)
    label_path = os.path.join(label_dir, mask_file.rsplit(".",1)[0]+".txt")


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue  

        poly = cnt.reshape(-1, 2)
        poly_norm = []
        for x, y in poly:
            poly_norm.append(x / w)
            poly_norm.append(y / h)

        line = "0 " + " ".join([f"{p:.6f}" for p in poly_norm])
        lines.append(line)

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print(label_dir)
