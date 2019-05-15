import dlib
from skimage import io
import cv2
import scipy
import os

'''#图片为一个人一张图片时
train_path="./data/dlib-align-images/"
test_path="./data/test/align/"
'''

train_path="./data/dlib-align-image/"
test_path="./data/test/align/"

def detect_face_from_img_path(image_path):
    # 使用 Dlib 的正面人脸检测器 frontal_face_detector
    detector = dlib.get_frontal_face_detector()

    img = io.imread(image_path)
    # 生成 Dlib 的图像窗口
    #为了框出检测到的人脸，用dlib.image_window()来加载显示窗口，
    # window.set_image(img)先将图片显示到窗口上，
    # 再利用window.add_overlay(dets)来绘制检测到的人脸框;
    win = dlib.image_window()
    win.set_image(img)
    # 使用detector检测器来检测图像中的人脸;faces的个数即为检测到的人脸的个数;遍历faces可以获取到检测到的每个人脸四个坐标极值。
    faces = detector(img, 1)#将检测器应用在输入图片上，结果返回给faces（参数1表示对图片进行上采样一次，有利于检测到更多的人脸）;
    print(type(faces[0]), '\n')

    print("人脸数 / faces in all：", len(faces))
    #enumerate的作用就是对可迭代的数据进行标号并将其里面的数据和标号一并打印出来
    for i, d in enumerate(faces):
        print("第", i+1, "个人脸的矩形框坐标：",
              "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(),'\t',  "bottom:", d.bottom())

    # 绘制矩阵轮廓
    win.add_overlay(faces)

    # 保持图像
    dlib.hit_enter_to_continue()


#predictor_path:人脸矫正模型对应的路径；face_file_path：被检测和矫正的图片对应的路径
def detect_alignment_face_from_img_path(predictor_path,face_file_path):
    path=os.path.split(face_file_path)#将路径和文件名分隔开
    pic_name=path[-1]#图片的名称
    path_file=face_file_path.split('/')
    file_name=path_file[-2]#图片所在的文件夹的名称
    #训练
    face_path = train_path + file_name + "/" + pic_name #处理后的图片存储路径
    #测试
    #face_path=test_path+pic_name
    #face_path='./data/dlib-align-images/'+file_name #定义人脸检测和人脸矫正后的图片存储路径
    detector = dlib.get_frontal_face_detector()#导入人脸检测器
    predictor = dlib.shape_predictor(predictor_path)#导入特征提取器

    img = io.imread(face_file_path)#读入图片
    #检测图片中的人脸
    dets = detector(img, 1)##将检测器应用在输入图片上，结果返回给dets（参数1表示对图片进行上采样一次，有利于检测到更多的人脸）
    print(type(dets))

    num_faces = len(dets)#检测图片上的人脸数量
    #print('人脸个数：',num_faces)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        return 0

    #window = dlib.image_window()  # 加载显示窗口
    #window.set_image(img)
    '''#控制台打印人脸在图片上的位置
    for i, d in enumerate(dets):
        print("第", i+1, "个人脸的矩形框坐标：",
              "left:", d.left(), '\t', "right:", d.right(), '\t', "top:", d.top(),'\t',  "bottom:", d.bottom())
    '''
    # 绘制矩阵轮廓
    #window.add_overlay(dets)
    #dlib.hit_enter_to_continue()

    #识别人脸特征点，并保存下来
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(predictor(img, detection))

    #window = dlib.image_window()#加载显示窗口

    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    #人脸对齐
    images = dlib.get_face_chips(img, faces, size=320)
    #显示对齐结果
    for image in images:
        #window.set_image(image)#将图片显示到窗口上
        scipy.misc.imsave(face_path, image) #将图片存储到本地
        #dlib.hit_enter_to_continue()
