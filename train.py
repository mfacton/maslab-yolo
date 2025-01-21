from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="block-dataset.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=(640, 480),  # training image size
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=-1,
)

# metrics = model.val()

# model.export(format="onnx")
