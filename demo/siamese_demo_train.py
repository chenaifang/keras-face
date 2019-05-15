import random

from keras_face.library.siamese import SiameseFaceNet
import os
import numpy as np

def main():
    fnet = SiameseFaceNet()
    '''
    因为VGGNET最后有三个全连接层, 因此,这个选项表示是否需要最上面的三个全连接层. 
    一般网络最后都会有全连接层, 最后一个全连接层更是设定了分类的个数, loss的计算方法, 
    并架设了一个概率转换函数(soft max). 其实soft max的作用就是将输出转换为各类别的概率,并计算loss. 
    可以这么说, 最上面三层是用来进行分类的, 其余层是用来进行特征提取的. 
    因此如果include_top=False,也就表示这个网络只能进行特征提取. 不能在进行新的训练或者在已有权重上fine-tune.
    '''
    #fnet.vgg16_include_top = True #包含最上层的全连接层
    #fnet.threshold=0.7  #阈值，在验证的时候作为是否是同一个人的标准
    model_dir_path = './model'
    #image_dir_path = "./data/dlib-align-images"
    #在测试
    image_dir_path = "./data/dlib-align-images"
    database = dict()
    #单人单张照片时
    '''
    for filename in os.listdir(r"./data/dlib-align-images"):
        list=filename.split(".")
        #print(list[0])
        database[list[0]]=[fnet.img_to_encoding(image_dir_path+"/"+filename)]
    '''
    '''#打算将模型整合起来
    for filename in os.listdir(r"./data/dlib-align-images-test"):
        pic_list = []#key是人名，value是两张图片的名字
        for picname in os.listdir(r"./data/dlib-align-images-test/" + filename):
            pic_list.append(fnet.img_to_encoding(image_dir_path + "/" + filename + "/" + picname))
            #pic_list.append(image_dir_path + "/" + filename + "/" + picname)
        database_name[filename] = pic_list

    fnet.fit(database1=database_name, model_dir_path=model_dir_path)
    '''

    #'''
    #i=0
    #j=0
    for filename in os.listdir(r"./data/dlib-align-images"):
        pic_list=[]
        for picname in os.listdir(r"./data/dlib-align-images/"+filename):
            pic_list.append(fnet.img_to_encoding(image_dir_path+"/"+filename+"/"+picname))
        database[filename]=pic_list

    np.save('picture_encoding.npy', database)
    #'''
    read_dictionary = np.load("picture_encoding.npy").item()
    #database=read_dictionary
    #fnet.fit(database=database, model_dir_path=model_dir_path)
        #i=i+1
        #if i is 17:
        #    j = j + 1
        #    np.save('picture_encoding_qqq.npy'+str(j), database)
        #    i = 0
        #    database=dict()
    #将字典保存到文件中
    #'''
    #'''
    #读取文件中的内容，将内容赋给字典
    #read_dictionary1 = np.load("picture_encoding1.npy").item()
    #read_dictionary2 = np.load("picture_encoding2.npy").item()
    #database=read_dictionary1.copy()
    #database.update(read_dictionary2

    #database=dict(read_dictionary1.items()+read_dictionary2.items())
    #print(database)
    #read_dictionary=database
    #random.shuffle(read_dictionary)
    database = read_dictionary

    #database是人名；
    fnet.fit(database=database, model_dir_path=model_dir_path)

    #'''

if __name__ == '__main__':
    main()