from ultralytics import YOLO

model = YOLO('./models/yolo11n.yaml')
model.info()

model = YOLO('./models/yolo100n.yaml')
model.info()

model = YOLO('./models/yolo101n.yaml')
model.info()

model = YOLO('./models/yolo102n.yaml')
model.info()

model = YOLO('./models/yolo103n.yaml')
model.info()

model = YOLO('./models/yolo104n.yaml')
model.info()
