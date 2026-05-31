import cv2
import os

os.makedirs('class_images', exist_ok=True)
cap = cv2.VideoCapture('test_classroom.mp4')
frame_count = 0
saved_count = 0

while cap.isOpened() and saved_count < 100:
    ret, frame = cap.read()
    if not ret: break
    # 每隔2帧保存一张，确保图片有一定多样性
    if frame_count % 5 == 0:
        cv2.imwrite(f'class_images/eval_{saved_count}.jpg', frame)
        saved_count += 1
    frame_count += 1
print(f"成功抽取 {saved_count} 张图片到 class_images 文件夹！")