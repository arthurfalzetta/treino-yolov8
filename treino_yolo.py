from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)

print("Fim do treinamento. Modelo salvo em runs/detect/train/weights/best.pt")