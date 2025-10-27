from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# results = model.train(data="coco8.yaml", epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()


# results = model.train(data="medical-pills.yaml", epochs=3, imgsz=640)

results = model.train(data="brain-tumor.yaml", epochs=10, imgsz=640)
results = model.val()