from ultralytics import YOLO

model = YOLO("runs/detect/train9/weights/best.pt")

results = model("datasets/images/test/4.jpg")
results[0].show()
