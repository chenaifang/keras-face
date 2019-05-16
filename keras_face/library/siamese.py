import random

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array
import os
import operator
import matplotlib.pyplot as pyplot
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image

#计算两张图片的欧氏距离
def euclidean_distance(vects):
    # x为Tensor("model_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
    # y为Tensor("model_1_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
    x, y = vects#x和y均是128维
    print(x)
    '''
    如果想在Keras中自定义各种层和函数，一定会用到的就是backend。一般导入的方法是from keras import backend as K
    这是因为Keras可以有两种后台，即theano和tensorflow，所以一些操作张量的函数可能是随后台的不同而不同的
    通过引入这个backend，就可以让Keras来处理兼容性。
    比如求x的平均，就是K.mean(x);sqrt()是平方根；maximum：逐位比较其最大者；square:平方
    backend文件本身在keras/backend文件夹下
    K.sum(K.square(x - y), axis=1, keepdims=True)
    axis=1 以竖轴为基准 ，同行相加
    keepdims主要用于保持矩阵的二维特性
    epsilon:返回数字表达式中使用的模糊因子的值。返回值是一个浮点数。
    '''
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#设置计算两张图片距离的输出维度；Lambda层的输出维度
def eucl_dist_output_shape(shapes):
    # shape1、shape2：(None, 128);  shape1[0],1:None 1   shapes:[(None, 128), (None, 128)];  shape1[0]:None  shape1[1]:128
    shape1, shape2 = shapes
    return (shape1[0], 1)

#对比损失
def contrastive_loss(y_true, y_pred):
    margin = 1 #自己设定的阈值
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


class SiameseFaceNet(object):
    model_name = 'siamese-face-net'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.vgg16_include_top = False
        self.labels = None
        self.config = None
        self.input_shape = (224,224,3)
        self.threshold = 0.5
        self.vgg16_model = None

    #完成图像的解码，大小处理，预处理，生成模型的输入
    def img_to_encoding(self, image_path):
        print('encoding: ', image_path)
        if self.vgg16_model is None:
            self.vgg16_model = self.create_vgg16_model()

        image = cv2.imread(image_path, 1)#加载图片原图形式
        #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 加载图片为灰度图像
        #print(type(image))
        #image=[image,image,image]
        #print(type(image))
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)#改变图片大小，修改为224*224
        input = img_to_array(img)#将图片转化为数组
        input = np.expand_dims(input, axis=0)#扩展数组的形状,input.shape为（1，224，224，3）
        input = preprocess_input(input)#完成数据预处理，对样本执行 逐样本均值消减 的归一化
        code_predict=self.vgg16_model.predict(input)
        base1_network = self.create_base1_network((7,7,512))
        co_predict=base1_network.predict(code_predict)
        return co_predict

    def create_base1_network(self, input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        return Model(input, x)

    def load_model(self, model_dir_path):
        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)#siamese-face-net-config.npy的存储路径
        #print(config_file_path)#./models\siamese-face-net-config.npy
        self.config = np.load(config_file_path).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        self.threshold = self.config['threshold']
        self.vgg16_include_top = self.config['vgg16_include_top']

        self.vgg16_model = self.create_vgg16_model()
        self.model = self.create_network(input_shape=self.input_shape)
        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_base_network(self, input_shape):
        '''Base network to be shared (eq. to feature extraction).
            要共享的基础网络（例如，特征提取）
        '''
        #Input()用于实例化Keras张量。  Keras张量是来自底层后端的张量对象
        input = Input(shape=input_shape) #Tensor("input_3:0", shape=(?, 1, 1000), dtype=float32)
        # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
        '''一个张量包含了一下几个信息
        一个名字，它用于键值对的存储，用于后续的检索：Const: 0
        一个形状描述， 描述数据的每一维度的元素个数：（2，3）
        数据类型，比如int32,float32
        '''
        x = Flatten()(input) #Tensor("flatten_1_1/Reshape:0", shape=(?, ?), dtype=float32)
        #Dense（全连接层）的两个参数：units：大于0的整数，代表该层的输出维度。activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
        x = Dense(1024, activation='relu')(x) #Tensor("dense_1/Relu:0", shape=(?, 128), dtype=float32)
        #dropout概率，输出的非0元素是原来的 “1/keep_prob” 倍
        x = Dropout(0.6)(x) #Tensor("dropout_1/cond/Merge:0", shape=(?, 128), dtype=float32)
        x = Dense(1024, activation='relu')(x) #Tensor("dense_2/Relu:0", shape=(?, 128), dtype=float32)
        x = Dropout(0.6)(x) #Tensor("dropout_2/cond/Merge:0", shape=(?, 128), dtype=float32)
        x = Dense(256, activation='relu')(x) #Tensor("dense_3/Relu:0", shape=(?, 128), dtype=float32)
        #input:Tensor("input_3:0", shape=(?, 1, 1000), dtype=float32);x:Tensor("dense_3/Relu:0", shape=(?, 128), dtype=float32)
        model=Model(input,x)
        plot_model(model, to_file='base_model.png', show_shapes=True)
        return Model(input, x)

    def accuracy(self, y_true, y_pred):
        #用固定的阈值计算距离的分类精度。
        #cast(x,dtype)改变张量的数据类型，返回Keras 张量，类型为 dtype；y_true的dtype是float32类型；y_pred<self.threshold是bool类型
        #mean：平均；equal：逐个元素对比两个张量的相等情况。
        return K.mean(K.equal(y_true, K.cast(y_pred < self.threshold, y_true.dtype)))

    def create_network(self, input_shape):
        # network definition  网络定义(好像是定义了输入层的大小以及随机失活和全连接层，以及最后一层全连接层的输出大小)
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)#Tensor("input_4:0", shape=(?, 1, 1000), dtype=float32)
        input_b = Input(shape=input_shape)#Tensor("input_5:0", shape=(?, 1, 1000), dtype=float32)

        #因为我们重新使用相同的实例`base_network`，所以网络的权重将在两个分支上共享
        processed_a = base_network(input_a)#Tensor("model_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
        processed_b = base_network(input_b)#Tensor("model_1_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
        '''
        新版本的Keras提供了Lambda层，以实现一些简单的计算任务
        如果你只是想对流经该层的数据做个变换，而这个变换本身没有什么需要学习的参数，那么直接用Lambda Layer是最合适的了。
        导入的方法是  from keras.layers.core import Lambda
        Lambda函数接受两个参数，第一个是输入张量对输出张量的映射函数，第二个是输入的shape对输出的shape的映射函数。
        第一个参数：要实现的函数；第二个参数：函数应该返回的值的shape；
        [processed_a, processed_b]是第一个函数参数的参数
        distance:Tensor("lambda_1/Sqrt:0", shape=(?, 1), dtype=float32)   
        distance是计算欧式距离     
        '''
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)#好像是输入是两张照片，输出是两张照片的欧式距离

        rms = RMSprop()#设置优化器，选择优化算法
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[self.accuracy])
        #print(model.summary())#输出模型各层的参数状况
        print("model summary")
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    #创建训练的数据对，则创建【人数*每人照片数】个数据对
    def create_pairs(self, database, names):
        '''Positive and negative pair creation.  创造正样本和负样本
        Alternates between positive and negative pairs.正负样本交替
        '''
        num_classes = len(database)#num_classes为参与训练的人的个数
        pairs = []
        labels = []

        n = min([len(database[name]) for name in database.keys()])#每个人对应的照片数量
        #每张图片依次处理;最后的输出为两个list，分别为图片（两张）
        for d in range(len(names)):
            name = names[d] #name为其中的一张图片人脸的名字
            x = database[name] #x是对应要处理图片名字的图片解码
            for i in range(n):
                pairs += [[x[i], x[(i + 1) % n]]]#x[0]和x是同一个人的照片
                inc = random.randrange(1, num_classes)#返回【1，num_classes）的一个随机数,即1至11中的一个随机数
                dn = (d + inc) % num_classes #dn和d不可能相同，即z1和z2不可能相同
                z1, z2 = x[i], database[names[dn]][i]
                pairs += [[z1, z2]]#在第一行加入了相同的数据对，在此加入了不同的数据对，即若两张图片相同，则对应标签是1，若两张图片不同，则对应标签是0
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        #model_dir_path:./model;os.path.sep:\;SiameseFaceNet.model_name:siamese-face-net
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-architecture.h5'

    def create_vgg16_model(self):
        #vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')#包含最上层的连接层，imagenet表示加载imagenet与训练的网络权重.
        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet',input_shape=(224,224,3))
        '''
        weights: ‘None’ / ‘imagenet’ / path (to the weight file) 
        None表示没有指定权重,对网络参数进行随机初始化. 
        'imagenet’ 表示加载imagenet与训练的网络权重. 
        ‘path’ 表示指向权重文件的路径. 
        '''
        '''
        在训练模型之前，需要通过compile来对学习过程进行配置。compile接收三个参数：
        优化器optimizer：指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象
        损失函数loss：最小化的目标函数，为预定义的损失函数名，如categorical_crossentropy、mse，也可以为一个损失函数
        指标列表metrics：对分类问题，一般设置为metrics=['accuracy']。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典.
        如果只是载入模型并利用其predict，可以不用进行compile。在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。predict会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数），
        
        SGD：随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量
        keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # lr：大或等于0的浮点数，学习率
        # momentum：大或等于0的浮点数，动量参数
        # decay：大或等于0的浮点数，每次更新后的学习率衰减值
        # nesterov：布尔值，确定是否使用Nesterov动量
        
        目标函数，或称损失函数，是编译一个模型必须的两个参数之一。
        可以通过传递预定义目标函数名字指定目标函数，也可以传递一个Theano/TensroFlow的符号函数作为目标函数，该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数：
        y_true：真实的数据标签，Theano/TensorFlow张量
        y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量
        真实的优化目标函数是在各个数据点得到的损失函数值之和的均值
        categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列。

        性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由metrics关键字设置。
        性能评估函数类似与目标函数, 只不过该性能的评估结果将不会用于训练.
        '''
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        vgg16_model.summary()
        plot_model(vgg16_model, to_file='vgg16_model.png', show_shapes=True)
        return vgg16_model

    # 绘制训练过程中的损失与精确度
    def training_vis(self,hist):
        loss = hist.history['loss']#loss
        val_loss = hist.history['val_loss']#val_loss
        acc = hist.history['accuracy']#accuracy
        val_acc = hist.history['val_accuracy']#val_accuracy

        # make a figure
        fig = plt.figure(figsize=(8, 4))
        print("figure")
        # subplot loss
        ax1 = fig.add_subplot(121)
        ax1.plot(loss, label='train_loss')
        ax1.plot(val_loss, label='val_loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss on Training and Validation Data')
        ax1.legend()
        # subplot acc
        ax2 = fig.add_subplot(122)
        ax2.plot(acc, label='train_acc')
        ax2.plot(val_acc, label='val_acc')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy  on Training and Validation Data')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def fit(self, database, model_dir_path, epochs=None, batch_size=None, threshold=None, vgg16_include_top=None):
        '''
        开始self.threshold为0.5；self.vgg16_include_top为True；
        经过几个if之后，如果threshold和vgg16_include_top设置了非空值，则将设置的值赋给threshold和vgg16_include_top
        如果batch_size为空值，则将其设置为128,；如果epochs为空值，则将其设置为20
        '''
        if threshold is not None:
            self.threshold = threshold
        if batch_size is None:
            batch_size = 512
        if epochs is None:
            epochs = 60
        if vgg16_include_top is not None:
            self.vgg16_include_top = vgg16_include_top
        #取database这个字典的第一个键值对
        #feature[0].shape为获取feature[0]的形状,feature[0]的大小为（1,1000），即1*1000
        #在keras中，数据是以张量的形式表示的，张量的形状称之为shape，表示从最外层向量逐步到达最底层向量的降维解包过程。
        #比如，一个一阶的张量[1,2,3]的shape是(3,); 一个二阶的张量[[1,2,3],[4,5,6]]的shape是(2,3); 一个三阶的张量[[[1],[2],[3]],[[4],[5],[6]]]的shape是(2,3,1)。
        #input_shape就是指输入张量的shape。
        for name, feature in database.items():
            self.input_shape = feature[0].shape
            break

        print("=======================")
        print(self.vgg16_include_top)
        self.vgg16_model = self.create_vgg16_model() #self.vgg16_model是在init函数中定义的
        self.model = self.create_network(input_shape=self.input_shape)#设置了模型的全连接层的输出维度并使用了随机失活，使用两张图片输入，输出为欧式距离
        architecture_file_path = self.get_architecture_path(model_dir_path)#设置siamese-face-net-architecture.h5的存储路径
        open(architecture_file_path, 'w').write(self.model.to_json())#将模型结构写入siamese-face-net-architecture.h5文件中,其中self.model的类型为<class 'keras.engine.training.Model'>

        names = [] #['danielle', 'younes', 'tian', 'andrew', 'kian']
        self.labels = dict() #设置self.labels为字典类型{'danielle': 0, 'younes': 1, 'tian': 2, 'andrew': 3, 'kian': 4}
        #主要是为了给self.labels赋值
        for name in database.keys():
            names.append(name)
            self.labels[name] = len(self.labels)
        self.config = dict()#设置self.config为字典 {'input_shape': (1, 1000), 'labels': {'danielle': 0, 'younes': 1, 'tian': 2, 'andrew': 3, 'kian': 4}, 'threshold': 0.5, 'vgg16_include_top': True}
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold
        self.config['vgg16_include_top'] = self.vgg16_include_top

        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)#存储模型的配置，siamese-face-net-config.npy
        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)#设置存储的权重的路径
        checkpoint = ModelCheckpoint(weight_file_path,period=1)# ModelCheckpoint 保存训练过程中的最佳模型权重

        #database是参与训练的每张照片，包括照片的编码；names是参与训练的照片的人名
        #t_y是一维矩阵，矩阵有24个数，t_y的shape为(24,)；t_x的shape为(24, 2, 1, 1000)
        t_x, t_y = self.create_pairs(database, names)

        print('data set pairs: ', t_x.shape)

        #print("model")
        #self.model.summary()
        #X[:,0]就是取所有行的第0个数据；X[:,1]就是取所有行的第1个数据
        hist=self.model.fit([t_x[:, 0], t_x[:, 1]], t_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.4,
                       verbose=SiameseFaceNet.VERBOSE,
                       callbacks=[checkpoint],
                       shuffle=True)
        #将训练过程中的损失与精确度保存
        with open('loss_and_acc.txt','w') as f:
            f.write(str(hist.history))

        self.training_vis(hist)

        self.model.save_weights(weight_file_path)

    def verify(self, image_path, identity, database, threshold=None):
        """
        验证“image_path”图像上的人是否为“身份”的函数。
        Arguments:
        image_path -- 图片路径
        identity --  字符串，您要验证身份的人的姓名。 必须是快乐屋的居民。
        database --将允许的人的姓名（字符串）的名称映射到他们的编码（向量）。
        model -- Keras中的Inception模型实例

        Returns:
        dist -- image_path与数据库中“identity”图像之间的距离。
        door_open -- 是的，如果门应该打开。 否则就错了。
        """
        #print("-----------")
        #print(self.threshold)
        #print(self.vgg16_include_top)
        if threshold is not None:
            self.threshold = threshold

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        #计算图像的编码。 使用img_to_encoding（）参见上面的例子。计算的是要对比的一张图片的编码
        encoding = self.img_to_encoding(image_path)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        #用身份的图像计算距离，identity是提前存在库里的图片,即要对比的另一张图片
        input_pairs = []#input_pairs是list类型,存的是两张图片的数字表达形式
        x = database[identity]#x是list，里面很多数，数据的类型为float32
        for i in range(len(x)):
            input_pairs.append([encoding, x[i]])

        input_pairs = np.array(input_pairs)#input_pairs类型为numpy.ndarray
        dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]
        print(dist)

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        #如果dist <阈值则打开门，否则不要打开
        print(self.threshold)
        if dist < self.threshold:
            print("It's " + str(identity))
            is_valid = True
        else:
            print("It's not " + str(identity))
            is_valid = False

        return dist, is_valid

    def who_is_it(self, image_path, database, threshold=None):
        """
        Implements face recognition for the happy house by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image 图片路径
        database -- database containing image encodings along with the name of the person on the image包含图像编码的数据库以及图像上人物的名称
        model -- your Inception model instance in Keras  Keras中的Inception模型实例

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
                     image_path编码与数据库编码之间的最小距离
        identity -- string, the name prediction for the person on image_path
        """

        if threshold is not None:
            self.threshold = threshold

        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
        encoding = self.img_to_encoding(image_path)

        ## Step 2: Find the closest encoding ##

        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, x) in database.items():

            input_pairs = []
            for i in range(len(x)):
                input_pairs.append([encoding, x[i]])
            input_pairs = np.array(input_pairs)
            dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]

            print("--for " + str(name) + ", the distance is " + str(dist))

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > self.threshold:
            print("Not in the database.")
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity

    
def main():
    fnet = SiameseFaceNet()
    
    #fnet.vgg16_include_top = True #包含最上层的全连接层
    #fnet.threshold=0.7  #阈值，在验证的时候作为是否是同一个人的标准
    model_dir_path = './model'
    #image_dir_path = "./data/dlib-align-images"
    #在测试
    image_dir_path = "./dlib-align-images4"
    database = dict()
    
    
    for filename in os.listdir(r"./dlib-align-images4"):
        pic_list=[]
        for picname in os.listdir(r"./dlib-align-images4/"+filename):
            pic_list.append(fnet.img_to_encoding(image_dir_path+"/"+filename+"/"+picname))
        database[filename]=pic_list
    
    np.save('picture_encoding4.npy', database)
    
    #read_dictionary = np.load("picture_encoding2.npy").item()
    
    #database = read_dictionary

    #database是人名；
    fnet.fit(database=database, model_dir_path=model_dir_path)

    print("result")

if __name__ == '__main__':
    main()
