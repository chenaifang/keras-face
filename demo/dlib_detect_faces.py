from keras_face.library.dlib_utils import detect_face_from_img_path,detect_alignment_face_from_img_path
import os

def main():

    #训练时处理的图片路径（一个人一张图片时）
    '''
    for filename in os.listdir(r"./data/imgs"):
        detect_alignment_face_from_img_path('./dlib-files/shape_predictor_68_face_landmarks.dat','./data/imgs/'+filename)
    '''

    #实际应用时处理的图片路径（一个人一张图片时）
    '''
    for filename in os.listdir(r"./data/test/images"):
        detect_alignment_face_from_img_path('./dlib-files/shape_predictor_68_face_landmarks.dat','./data/test/images/'+filename)
    '''

    #训练时处理的图片路径（一个人两张图片）
    #'''
    i=0
    for filename in os.listdir(r"./data/images"):
        os.mkdir("./data/dlib-align-image/" + filename)
        for pic in os.listdir(r"./data/images/"+filename):
            detect_alignment_face_from_img_path('./dlib-files/shape_predictor_68_face_landmarks.dat','./data/images/' + filename+ "/" + pic)
            i = i + 1
            if i is 2:
                i=0
                break
    #'''
    '''#明星图片文件夹已删
    #训练时处理自己搜集到的图片
    for filename in os.listdir(r"./data/明星图片"):
        os.mkdir("./data/矫正/" + filename)
        for pic in os.listdir(r"./data/明星图片/"+filename):
            detect_alignment_face_from_img_path('./dlib-files/shape_predictor_68_face_landmarks.dat','./data/明星图片/' + filename+ "/" + pic)
    '''
    #处理单人的多张照片
    '''
    for filename in os.listdir(r"./data/images/Yasar_Yakis"):
        detect_alignment_face_from_img_path('./dlib-files/shape_predictor_68_face_landmarks.dat',
                                            './data/images/Yasar_Yakis/' + filename)
    '''
if __name__ == '__main__':
    main()
