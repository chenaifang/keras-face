import numpy as np
import scipy.io as sio

from demo import vgg16
from . import *
import tensorflow as tf
import utils
import cv2
import numpy as np
import skimage
from skimage import io
from skimage import transform
import pandas as pd
from scipy.linalg import norm
import os


'''
batch_size = 32
annotation_path = './data/dlib-align-images-test/'
flickr_image_path = 'D:/dataset_code/数据集/flickr+mscoco/flickr30k/flickr30k-images/'
feat_path = './weight/vgg16_feats.npy'
annotation_result_path = './data/dlib-align-images-test/annotations.pickle'
annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path, x.split('#')[0]))

#获取文件夹下每一张图片
unique_images = annotations['image'].unique()
#print(len(unique_images))#31783
image_df = pd.DataFrame({'image': unique_images, 'image_id': range(len(unique_images))})
# 每张图片对应5个句子
annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)
'''
pic_list = []#key是人名，value是两张图片的名字
for filename in os.listdir(r"./data/dlib-align-images-test"):
    for picname in os.listdir(r"./data/dlib-align-images-test/" + filename):
        pic_list.append("./data/dlib-align-images-test" + "/" + filename + "/" + picname)
        #pic_list.append(image_dir_path + "/" + filename + "/" + picname)

unique_images = pic_list
def get_feats():
    vgg16_feats = np.zeros((len(unique_images), 4096))
    with tf.Session() as sess:
        images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        vgg = vgg16.Vgg16()
        vgg.build(images)
        for i in range(len(unique_images)):
            img_list = utils.load_image(unique_images[i])
            batch = img_list.reshape((1, 224, 224, 3))
            feature = sess.run(vgg.fc7, feed_dict={images: batch})#提取fc7层的特征
            feature = np.reshape(feature, [4096])
            feature /= norm(feature) # 特征归一化
            vgg16_feats[i, :] = feature #每张图片的特征向量为1行
    vgg16_feats = np.save('./vgg16_feats', vgg16_feats)
    return vgg16_feats



if __name__ == '__main__':
    get_feats()