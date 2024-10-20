from ultralytics import YOLO

model = YOLO('weights/best.pt')

model.export(
    format='engine', 
    device='cuda',
    batch=1,
    workspace=4,
    int8=False,
    half=True,
    )

