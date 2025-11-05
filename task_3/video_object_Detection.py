from ultralytics import YOLO
import os
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "yolov11n-seg.pt" 
INPUT_FOLDER = "input_frames"
OUTPUT_FOLDER = "processed_frames"
DETECTED_FOLDER = "output_frames/run1"  


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)])

frame_indices = []
avg_confidences = []
detections_per_frame = []
seg_area_ratios = []

for idx, img_name in enumerate(tqdm(image_files, desc="Processing frames")):
    input_path = os.path.join(INPUT_FOLDER, img_name)
    output_path = os.path.join(OUTPUT_FOLDER, img_name)


    results = model.predict(
        source=input_path,
        device=device,
        save=True,                
        project="output_frames",    
        name="run1",                 
        exist_ok=True,
        verbose=False,
        conf=0.5
    )

    results[0].save(filename=output_path)

    boxes = results[0].boxes
    masks = results[0].masks

    confs = [float(b.conf) for b in boxes] if boxes is not None else []
    avg_conf = np.mean(confs) if confs else 0
    num_detections = len(confs)

    if masks is not None and masks.data is not None:
        mask_area = float(masks.data.sum())  
        total_area = masks.data.numel()     
        seg_ratio = mask_area / total_area
    else:
        seg_ratio = 0


    frame_indices.append(idx + 1)
    avg_confidences.append(avg_conf)
    detections_per_frame.append(num_detections)
    seg_area_ratios.append(seg_ratio)



