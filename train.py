from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="block-dataset.yaml",
    epochs=100,
    imgsz=(640, 480),
    device="cuda",
    batch=-1,
)

# metrics = model.val()

# model.export(format="onnx")
