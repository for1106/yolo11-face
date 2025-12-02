from ultralytics import YOLO, settings

root = '/content/drive/MyDrive/yolo'
root = '.'

settings.update({
    'runs_dir': f'{root}/',
    'datasets_dir': f'{root}/dataset/widerface'
})

model = YOLO(f'{root}/models/yolo104n.yaml')
results = model.train(
    data=f'{root}/widerface.yaml',
    device='mps',
    imgsz=640,
    epochs=100,
    batch=16,
    optimizer='SGD',
    max_det=300
)

# model = YOLO(f'{root}/detect/train/weights/last.pt')
# results = model.train(resume=True)
