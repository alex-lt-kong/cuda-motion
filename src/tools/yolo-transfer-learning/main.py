from ultralytics import YOLOWorld

# Load a YOLO-World model (e.g., yolov8s-world.pt)
model = YOLOWorld('yolov8s-worldv2.pt')

# Tell it what to look for (No training required!)
model.set_classes(["crib", "baby"])

# Run inference
results = model.predict('your_stream.mp4')