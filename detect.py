import time
import cv2
import numpy as np
from ultralytics import YOLO
from pytubefix import YouTube


def if_youtube(url):
    if not isinstance(url, str):
        return url

    if 'youtube.com' in url:
        yt = YouTube(url)
        stream = yt.streams.filter(
            progressive=True,
            file_extension='mp4'
        ).order_by('resolution').desc().first()
        return stream.url

    return url


def run_model(frame, model, color, conf=0.5):
    img = frame.copy()
    start = time.time()
    results = model(img, verbose=False, conf=conf)
    end = time.time()

    count = 0
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            count += 1

    return img, count, end - start


def draw_label(frame, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (255, 255, 255), -1)
    cv2.putText(frame, text, (15, 15 + th),
                font, scale, color, thickness, cv2.LINE_AA)


def combine_frames(imgs):
    n = len(imgs)

    if n == 1:
        # 只有一張，直接用它
        return imgs[0]

    elif n == 2:
        # 兩張，上下堆疊
        return np.vstack((imgs[0], imgs[1]))

    elif n == 4:
        # 四張：先左右，再上下
        top = np.hstack((imgs[0], imgs[1]))
        bottom = np.hstack((imgs[2], imgs[3]))
        return np.vstack((top, bottom))

    elif n == 6:
        # 六張：先左右，再上下
        top = np.hstack((imgs[0], imgs[1]))
        middle = np.hstack((imgs[2], imgs[3]))
        bottom = np.hstack((imgs[4], imgs[5]))
        return np.vstack((top, middle, bottom))

    else:
        return imgs[0]


def main(input_source):
    input_source = if_youtube(input_source)

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'width: {width}, height: {height}')

    # 外部模型
    yolo11m = YOLO('./models/yolo11m.pt')

    # 我的模型
    model1 = YOLO('./detect/train1/weights/best.pt')
    # 更動資料集
    model2 = YOLO('./detect/train2/weights/best.pt')
    model3 = YOLO('./detect/train3/weights/best.pt')
    # 更動模型 320
    model100 = YOLO('./detect/train4/weights/best.pt')
    model101 = YOLO('./detect/train5/weights/best.pt')
    model102 = YOLO('./detect/train6/weights/best.pt')
    model103 = YOLO('./detect/train7/weights/best.pt')
    model104 = YOLO('./detect/train8/weights/best.pt')
    # 更動模型 640
    colab11 = YOLO('./detect/train9/weights/best.pt')
    colab103 = YOLO('./detect/train10/weights/best.pt')
    colab104 = YOLO('./detect/train11/weights/best.pt')

    model_map = [
        ('YOLO', yolo11m),

        # 320
        # ('YOLO-320', model1),
        # ('100',  model100),
        # ('101',  model101),
        # ('102',  model102),
        # ('103',  model103),
        # ('104',  model104),

        # 640
        # ('YOLO-640',  colab11),
        # ('colab103',  colab103),
        # ('colab104',  colab104),
    ]

    colors = [
        (0, 165, 255),   # 橘
        (0, 0, 255),     # 紅
        (255, 0, 255),   # 紫
        (0, 255, 0),     # 綠
        (0, 255, 0),     # 綠
        (0, 255, 0),     # 綠
    ]
    frame_count = 0
    box_count = [0, 0, 0, 0, 0, 0]
    detect_time = [0, 0, 0, 0, 0, 0]
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if input_source.startswith(('http://', 'https://', 'rtsp://')):
                    cap.release()
                    cap = cv2.VideoCapture(input_source)
                    continue
                else:
                    paused = not paused
                    continue
            frame_count += 1
            imgs = []
            for i, (name, model) in enumerate(model_map):
                img, c, t = run_model(frame, model, colors[i])

                box_count[i] += c
                detect_time[i] += t

                label = f"{name} - {box_count[i]}(box) - {(detect_time[i]/frame_count)*1000:.2f}(ms)"
                draw_label(img, label, colors[i])

                imgs.append(img)

            frame = combine_frames(imgs)

            cv2.imshow('frame', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # INPUT = 0
    # INPUT = './dataset/video/video.mp4'
    # INPUT = 'https://www.youtube.com/watch?v=_CrTi1aNJ-E'
    # INPUT = 'https://www.youtube.com/watch?v=-uzuhqQIaTM'
    INPUT = 'https://tcnvr3.taichung.gov.tw/c3c46934'
    # INPUT = 'https://tcnvr7.taichung.gov.tw/4f63f1ac'
    # INPUT = 'https://cctv-ss03.thb.gov.tw/T3-187K+900'
    main(INPUT)
