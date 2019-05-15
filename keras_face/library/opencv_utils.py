import numpy as np
import cv2
import os


def detect_face_from_img_path(frontal_face_model_file_path,frontal_eye_model_file_path,image_path):
    if not os.path.exists(frontal_face_model_file_path):
        print('failed to find face detection opencv model: ', frontal_face_model_file_path)

    if not os.path.exists(frontal_eye_model_file_path):
        print('failed to find eye detection opencv model: ', frontal_eye_model_file_path)

    face_cascade = cv2.CascadeClassifier(frontal_face_model_file_path)
    eye_cascade = cv2.CascadeClassifier(frontal_eye_model_file_path)
    face_cascade.load(
        'F:/Graduation Project/GraduationProject/Medium term/keras-face-master/demo/opencv-files/haarcascade_frontalface_alt.xml')
    eye_cascade.load(
        'F:/Graduation Project/GraduationProject/Medium term/keras-face-master/demo/opencv-files/haarcascade_eye.xml')

    img = cv2.imread(image_path)#读取图片，忽略alpha通道
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转换为灰度图像

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)#grayΪҪ��������ͼ��scaleFactor��ʾÿ��ͼ��ߴ��С�ı�����minNeighbors��ʾÿһ��Ŀ������Ҫ����⵽5�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽����),
    print('faces detected: ', len(faces))
    #框人脸
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print('eyes detected: ',len(eyes))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    frontal_face_model_file_path = '../../demo/opencv-files/haarcascade_frontalface_alt.xml'

    detect_face_from_img_path(
        frontal_face_model_file_path,
        '../../demo/data/opencv-images/test1.jpg')

    detect_face_from_img_path(
        frontal_face_model_file_path,
        '../../demo/data/opencv-images/test4.jpg')


if __name__ == '__main__':
    main()
