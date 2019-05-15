from keras_face.library.opencv_utils import detect_face_from_img_path


def main():
    frontal_face_model_file_path = './opencv-files/haarcascade_frontalface_alt.xml'
    frontal_eye_model_file_path = './opencv-files/haarcascade_eye.xml'



    detect_face_from_img_path(
        frontal_face_model_file_path,frontal_eye_model_file_path,
        './data/opencv-images/test1.jpg')

    detect_face_from_img_path(
        frontal_face_model_file_path,frontal_eye_model_file_path,
        './data/opencv-images/test4.jpg')

    detect_face_from_img_path(
        frontal_face_model_file_path,frontal_eye_model_file_path,
        './data/opencv-images/12.jpg')

if __name__ == '__main__':
    main()

