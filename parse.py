import os
import cv2
from tqdm import tqdm


def getPath(root, split):
    label_file = f'{root}/dataset/wider_face_split/wider_face_{split}_bbx_gt.txt'
    img_root = f'{root}/dataset/WIDER_{split}/images'
    save_root = f'{root}/dataset/widerface/{split}'
    return label_file, img_root, save_root


def main(path_tuple, skip):
    label_file, img_root, save_root = path_tuple
    os.makedirs(save_root, exist_ok=True)

    with open(label_file, 'r') as f:
        lines = f.read().splitlines()

    img_path = ''
    labels = []
    is_face2 = False

    info = {
        'count': 0,
        'num': 0,
        'num_used': 0,
        'face_0': 0,  # 0
        'face_1': 0,  # 1~30
        'face_2': 0,  # 30~
        'blur_0': 0,
        'blur_1': 0,
        'blur_2': 0,
        'invalid_0': 0,  # 易分辨
        'invalid_1': 0,  # 難分辨
        'occlusion_0': 0,  # ~1%
        'occlusion_1': 0,  # 1% - 30%
        'occlusion_2': 0,  # 30%~
    }

    def save_current():
        nonlocal is_face2

        if is_face2:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            for x, y, bw, bh in labels:
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w - 1, int(x + bw))
                y2 = min(h - 1, int(y + bh))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('image', img)
            key = cv2.waitKey(0)

            if key == 27:
                cv2.destroyAllWindows()
                exit(0)

            is_face2 = False

        if not img_path or not labels:
            return

        info['count'] += 1

        # 只是要統計
        if skip is True:
            return

        img = cv2.imread(img_path)
        if img is None:
            print('skip:', img_path)
            return

        h, w = img.shape[:2]

        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(save_root, f'{base}.txt')
        jpg_path = os.path.join(save_root, f'{base}.jpg')

        cv2.imwrite(jpg_path, img)

        # 去重 labels（假設 labels 是 [(x, y, bw, bh), ...]）
        unique_labels = []
        seen = set()
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        with open(txt_path, 'w') as f:
            for x, y, bw, bh in unique_labels:
                # YOLO 格式：xc, yc, w, h (normalized)
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                # class 固定為 0（face）
                f.write(f'0 {xc} {yc} {bw} {bh}\n')

    progress_bar = tqdm(lines)
    for line in progress_bar:
        line = line.strip()
        if line.lower().endswith('.jpg'):
            save_current()
            img_path = os.path.join(img_root, line)
            labels = []
        else:
            nums = list(map(float, line.split()))
            if len(nums) >= 4:
                x, y, bw, bh = nums[:4]

                blur = int(nums[4])
                invalid = int(nums[7])
                occlusion = int(nums[8])

                # 在此過濾資料
                # if bw < 10 or bh < 10:
                #     continue
                # if blur == 1 or blur == 2:
                #     continue
                # if invalid == 1:
                #     continue
                # if occlusion == 2:
                #     continue

                if bw > 0 and bh > 0:
                    labels.append((x, y, bw, bh))
                    info['num_used'] += 1

                info[f'blur_{blur}'] += 1
                info[f'invalid_{invalid}'] += 1
                info[f'occlusion_{occlusion}'] += 1
            else:
                num = int(line)
                info['num'] += num
                if num == 0:
                    info['face_0'] += 1
                elif num <= 30:
                    info['face_1'] += 1
                else:
                    info['face_2'] += 1
                    # 放置於要顯示的條件下
                    # is_face2 = True

    save_current()
    print(info)


if __name__ == '__main__':
    # 開啟則會自動將圖片數據轉換入訓練資料夾
    skip = True

    # colab or localhost
    root = '/content/drive/MyDrive/yolo'
    root = '.'

    main(getPath(root, 'train'), skip)
    main(getPath(root, 'val'), skip)
