from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt') 

results = model.train(
    data='/home/sashreekkumar/Documents/IIITH-AI-ML-Internship/kvasir/data.yaml',
    epochs=200,            
    imgsz=640,              
    batch=16,               
    device=0,              
    workers=4,              
    project='polyp_seg',     
    name='yolo11n_run',      
    patience=50,           
    save=True,             
    plots=True,             
    val=True,              
)


print("Training completed!")
print(f"Results saved to: {results.save_dir}")
print(f"Best model: {model.trainer.best}")

print("Validating best model...")
metrics = model.val()
print(f"Validation mAP50: {metrics.seg.map50}")
print(f"Validation mAP50-95: {metrics.seg.map}")

print("Exporting model to ONNX format...")
model.export(format='onnx')  