from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt') 

# Train the model
results = model.train(
    data='/home/sashreekkumar/Documents/IIITH-AI-ML-Internship/kvasir/data.yaml',
    epochs=200,              # More epochs for small dataset
    imgsz=640,               # Image size (640x640) - good balance for varied resolutions
    batch=16,                # Batch size (adjust based on GPU memory)
    device=0,                # Use GPU 0 (use 'cpu' for CPU training)
    workers=4,               # Reduced workers for small dataset
    project='polyp_seg',     # Project name
    name='yolo11n_run',      # Experiment name
    patience=50,             # Early stopping patience
    save=True,               # Save checkpoints
    plots=True,              # Save training plots
    val=True,                # Validate during training
)

# Print training results
print("\nTraining completed!")
print(f"Results saved to: {results.save_dir}")
print(f"Best model: {model.trainer.best}")

# Validate the best model
print("\nValidating best model...")
metrics = model.val()
print(f"Validation mAP50: {metrics.seg.map50}")
print(f"Validation mAP50-95: {metrics.seg.map}")

# Export the model 
print("\nExporting model to ONNX format...")
model.export(format='onnx')  