from keras_face.library.siamese import SiameseFaceNet


def main():
    fnet = SiameseFaceNet()

    model_dir_path = './model'
    image_dir_path = "./data/test/align"
    fnet.load_model(model_dir_path)

    database1 = dict() #dictionary


    #fnet.who_is_it(image_dir_path + "/younes.jpg", database)
    database1["chenaifang"]=[fnet.img_to_encoding(image_dir_path + "/bailu1.jpg")]
    fnet.verify(image_dir_path + "/wangtiange1.jpg", "chenaifang", database1)
    #print(fnet.verify(image_dir_path + "/tian.jpg", "younes", database))

if __name__ == '__main__':
    main()