from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=r"E:/QRLPR/t_2.2/anpr-yolov8/models_0/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8/data.yaml", epochs=2)  # train the model