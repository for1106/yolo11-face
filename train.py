from ultralytics import YOLO, settings

# colab or localhost
root = '/content/drive/MyDrive/yolo'
root = '.'

# 初始訓練
settings.update({
    'runs_dir': f'{root}/',
    'datasets_dir': f'{root}/dataset/widerface'
})

model = YOLO(f'{root}/models/yolo11n.yaml')
results = model.train(
    data=f'{root}/widerface.yaml',
    device=0,
    imgsz=640,
    epochs=100,
    batch=16,
    optimizer='SGD',
    max_det=30
)

# 繼續訓練
# model = YOLO(f'{root}/detect/train/weights/last.pt')
# results = model.train(resume=True)
