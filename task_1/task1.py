from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Train and validate
model.train(data="coco8.yaml", epochs=3)
model.val()

# Run detection
results = model("https://ultralytics.com/images/bus.jpg")

# Export to ONNX
model.export(format="onnx")

# Load exported model
onnx_model = YOLO("/home/sashreekkumar/Documents/runs/detect/train2/weights/best.onnx")

# Inference
onnx_results = onnx_model("https://ultralytics.com/images/bus.jpg")

for r in onnx_results:
    r.show()
    r.save()
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        print(f"Class: {r.names[int(cls)]}, Confidence: {conf:.2f}, Box: {box.tolist()}")

# Segmentation
seg_model = YOLO("yolo11n-seg.pt")
seg_results = seg_model("https://ultralytics.com/images/bus.jpg")

for r in seg_results:
    r.show()
    r.save()
