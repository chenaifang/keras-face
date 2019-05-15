import os
import shutil

'''#最开始的单人单张照片的处理
for filename in os.listdir(r"./data/lfw-deepfunneled"):
    #print(filename)
    for pic in os.listdir(r"./data/lfw-deepfunneled/"+filename):
        #print(pic)
        list=pic.split(".")
        last=list[-1]
        shutil.copy("./data/lfw-deepfunneled/"+filename+"/"+pic,"./data/imgs/"+filename+"."+last)
        break
'''
#修改为一个人两张照片后的代码
for filename in os.listdir(r"./data/lfw-deepfunneled"):
    #print(filename)
    pic_num=0
    os.mkdir("./data/images/" + filename)
    for pic in os.listdir(r"./data/lfw-deepfunneled/"+filename):
        length = len([lists for lists in os.listdir(r"./data/lfw-deepfunneled/"+filename)])
        #print(length)
        #print(pic)
        if length is 1:
            list=pic.split(".")
            last=list[-1]#后缀名，即照片格式
            shutil.copy("./data/lfw-deepfunneled/" + filename + "/" + pic, "./data/images/" + filename + "/" + pic)
            shutil.copy("./data/lfw-deepfunneled/" + filename + "/" + pic, "./data/images/" + filename + "/" + filename + "_0002" + "." + last)
            break
        if length is not 1:
            list = pic.split(".")
            last = list[-1]  # 后缀名，即照片格式
            pic_num=pic_num+1
            shutil.copy("./data/lfw-deepfunneled/" + filename + "/" + pic, "./data/images/" + filename + "/" + pic)
            if pic_num is 2:
                break
