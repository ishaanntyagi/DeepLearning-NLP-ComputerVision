from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data = r"C:\Users\ishaan.narayan\Downloads\Fracture.v1i.yolov8\data.yaml",
    epochs = 50, 
    imgsz = 640,
    batch = 8,
    project = 'Fracture_yolo_training',
    name = 'exp1')
