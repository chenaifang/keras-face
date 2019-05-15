from keras import backend as K

K.set_image_data_format('channels_first')
from keras_face.library.fr_utils import *
from keras_face.library.inception_blocks_v2 import *
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
import random
from keras.callbacks import ModelCheckpoint

# 计算两张图片的欧氏距离
def euclidean_distance(vects):
    # x为Tensor("model_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
    # y为Tensor("model_1_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
    x, y = vects  # x和y均是128维
    print(x)
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# 设置计算两张图片距离的输出维度；Lambda层的输出维度
def eucl_dist_output_shape(shapes):
    # shape1、shape2：(None, 128);  shape1[0],1:None 1   shapes:[(None, 128), (None, 128)];  shape1[0]:None  shape1[1]:128
    shape1, shape2 = shapes
    return (shape1[0], 1)

#对比损失
def contrastive_loss(y_true, y_pred):
    margin = 1 #自己设定的阈值
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = K.sum(K.maximum(basic_loss, 0))

    return loss

class SiameseFaceNet(object):
    model_name = 'siamese-face-net'
    VERBOSE = 1

    def __init__(self):
        self.model = None
        self.labels = None
        self.config = None
        self.input_shape = None
        self.threshold = 0.5
        self.facenet_model = None

    # 完成图像的解码，大小处理，预处理，生成模型的输入
    def img_to_encoding(self, image_path,model_dir_path):
        print('encoding: ', image_path)
        if self.facenet_model is None:
            self.facenet_model = self.create_facenet_model(model_dir_path)
        return img_to_encoding(image_path, self.facenet_model)

    def load_model(self, model_dir_path):
        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)  # siamese-face-net-config.npy的存储路径
        # print(config_file_path)#./models\siamese-face-net-config.npy
        self.config = np.load(config_file_path).item()
        self.labels = self.config['labels']
        self.input_shape = self.config['input_shape']
        #self.threshold = self.config['threshold']
        #model1_dir_path = model_dir_path
        self.facenet_model = self.create_facenet_model(model_dir_path)
        self.model = self.create_network(input_shape=self.input_shape)
        #weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)
        #self.model.load_weights(weight_file_path)

    def create_facenet_model(self,model_dir_path):
        FRmodel = faceRecoModel(input_shape=(3, 96, 96)) # 包含最上层的连接层，imagenet表示加载imagenet与训练的网络权重.
        print("Total Params:", FRmodel.count_params())
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(FRmodel, model_dir_path)
        FRmodel.summary()
        plot_model(FRmodel, to_file='facenet_model.png', show_shapes=True)
        return FRmodel

    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)  # Tensor("input_3:0", shape=(?, 1, 1000), dtype=float32)

        x = Flatten()(input)  # Tensor("flatten_1_1/Reshape:0", shape=(?, ?), dtype=float32)
        # Dense（全连接层）的两个参数：units：大于0的整数，代表该层的输出维度。activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
        x = Dense(128, activation='relu')(x)  # Tensor("dense_1/Relu:0", shape=(?, 128), dtype=float32)
        # dropout概率，输出的非0元素是原来的 “1/keep_prob” 倍
        x = Dropout(0.5)(x)  # Tensor("dropout_1/cond/Merge:0", shape=(?, 128), dtype=float32)
        x = Dense(128, activation='relu')(x)  # Tensor("dense_2/Relu:0", shape=(?, 128), dtype=float32)
        x = Dropout(0.5)(x)  # Tensor("dropout_2/cond/Merge:0", shape=(?, 128), dtype=float32)
        x = Dense(128, activation='relu')(x)  # Tensor("dense_3/Relu:0", shape=(?, 128), dtype=float32)
        # input:Tensor("input_3:0", shape=(?, 1, 1000), dtype=float32);x:Tensor("dense_3/Relu:0", shape=(?, 128), dtype=float32)
        return Model(input, x)

    def create_network(self, input_shape):
        # network definition  网络定义(好像是定义了输入层的大小以及随机失活和全连接层，以及最后一层全连接层的输出大小)
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)  # Tensor("input_4:0", shape=(?, 1, 1000), dtype=float32)
        input_b = Input(shape=input_shape)  # Tensor("input_5:0", shape=(?, 1, 1000), dtype=float32)

        # 因为我们重新使用相同的实例`base_network`，所以网络的权重将在两个分支上共享
        processed_a = base_network(input_a)  # Tensor("model_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
        processed_b = base_network(input_b)  # Tensor("model_1_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
        #base_network = faceRecoModel(input_shape=(3, 96, 96))
        #load_weights_from_FaceNet(base_network, './models')
        #processed_a = base_network(input_shape=(1, 3, 96, 96))  # Tensor("model_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)
        #processed_b = base_network(input_shape=(1, 3, 96, 96))  # Tensor("model_1_1/dense_3/Relu:0", shape=(?, 128), dtype=float32)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)  # 好像是输入是两张照片，输出是两张照片的欧式距离

        rms = RMSprop()  # 设置优化器，选择优化算法
        model.compile(loss=contrastive_loss, optimizer='adam', metrics=[self.accuracy])
        #model.compile(loss=contrastive_loss, optimizer=rms, metrics=[self.accuracy])
        # print(model.summary())#输出模型各层的参数状况
        print("model summary")
        model.summary()
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def accuracy(self, y_true, y_pred):
        # 用固定的阈值计算距离的分类精度。
        # cast(x,dtype)改变张量的数据类型，返回Keras 张量，类型为 dtype；y_true的dtype是float32类型；y_pred<self.threshold是bool类型
        # mean：平均；equal：逐个元素对比两个张量的相等情况。
        return K.mean(K.equal(y_true, K.cast(y_pred < self.threshold, y_true.dtype)))


    # 创建训练的数据对，则创建【人数*每人照片数】个数据对
    def create_pairs(self, database, names):
        '''Positive and negative pair creation.  创造正样本和负样本
        Alternates between positive and negative pairs.正负样本交替
        '''
        num_classes = len(database)  # num_classes为参与训练的人的个数
        pairs = []
        labels = []

        n = min([len(database[name]) for name in database.keys()])  # 每个人对应的照片数量
        # 每张图片依次处理;最后的输出为两个list，分别为图片（两张）
        for d in range(len(names)):
            name = names[d]  # name为其中的一张图片人脸的名字
            x = database[name]  # x是对应要处理图片名字的图片解码
            for i in range(n):
                pairs += [[x[i], x[(i + 1) % n]]]  # x[0]和x是同一个人的照片
                inc = random.randrange(1, num_classes)  # 返回【1，num_classes）的一个随机数,即1至11中的一个随机数
                dn = (d + inc) % num_classes  # dn和d不可能相同，即z1和z2不可能相同
                z1, z2 = x[i], database[names[dn]][i]
                pairs += [[z1, z2]]  # 在第一行加入了相同的数据对，在此加入了不同的数据对，即若两张图片相同，则对应标签是1，若两张图片不同，则对应标签是0
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + os.path.sep + 'siamese-face-net-config.npy'

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-facenet-weights.h5'

    @staticmethod
    def get_architecture_path(model_dir_path):
        # model_dir_path:./model;os.path.sep:\;SiameseFaceNet.model_name:siamese-face-net
        return model_dir_path + os.path.sep + SiameseFaceNet.model_name + '-facenet-architecture.h5'



    # 绘制训练过程中的损失与精确度
    def training_vis(self, hist):
        loss = hist.history['loss']  # loss
        val_loss = hist.history['val_loss']  # val_loss
        acc = hist.history['accuracy']  # accuracy
        val_acc = hist.history['val_accuracy']  # val_accuracy

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

    def fit(self, database, model_dir_path, epochs=None, batch_size=None, threshold=None):
        '''
        开始self.threshold为0.5；self.vgg16_include_top为True；
        经过几个if之后，如果threshold和vgg16_include_top设置了非空值，则将设置的值赋给threshold和vgg16_include_top
        如果batch_size为空值，则将其设置为128,；如果epochs为空值，则将其设置为20
        '''

        if threshold is not None:
            self.threshold = threshold
        if batch_size is None:
            batch_size = 256
        if epochs is None:
            epochs = 50
        # 取database这个字典的第一个键值对
        # feature[0].shape为获取feature[0]的形状,feature[0]的大小为（1,1000），即1*1000
        # 在keras中，数据是以张量的形式表示的，张量的形状称之为shape，表示从最外层向量逐步到达最底层向量的降维解包过程。
        # 比如，一个一阶的张量[1,2,3]的shape是(3,); 一个二阶的张量[[1,2,3],[4,5,6]]的shape是(2,3); 一个三阶的张量[[[1],[2],[3]],[[4],[5],[6]]]的shape是(2,3,1)。
        # input_shape就是指输入张量的shape。
        for name, feature in database.items():
            self.input_shape = feature[0].shape
            break

        #self.facenet_model = self.create_facenet_model(model_dir_path)  # self.vgg16_model是在init函数中定义的
        self.model = self.create_network(input_shape=self.input_shape)  # 设置了模型的全连接层的输出维度并使用了随机失活，使用两张图片输入，输出为欧式距离
        architecture_file_path = self.get_architecture_path(model_dir_path)  # 设置siamese-face-net-architecture.h5的存储路径
        open(architecture_file_path, 'w').write(
            self.model.to_json())  # 将模型结构写入siamese-face-net-architecture.h5文件中,其中self.model的类型为<class 'keras.engine.training.Model'>

        names = []  # ['danielle', 'younes', 'tian', 'andrew', 'kian']
        self.labels = dict()  # 设置self.labels为字典类型{'danielle': 0, 'younes': 1, 'tian': 2, 'andrew': 3, 'kian': 4}
        # 主要是为了给self.labels赋值
        for name in database.keys():
            names.append(name)
            self.labels[name] = len(self.labels)
        self.config = dict()  # 设置self.config为字典 {'input_shape': (1, 1000), 'labels': {'danielle': 0, 'younes': 1, 'tian': 2, 'andrew': 3, 'kian': 4}, 'threshold': 0.5, 'vgg16_include_top': True}
        self.config['input_shape'] = self.input_shape
        self.config['labels'] = self.labels
        self.config['threshold'] = self.threshold

        config_file_path = SiameseFaceNet.get_config_path(model_dir_path=model_dir_path)
        np.save(config_file_path, self.config)  # 存储模型的配置，siamese-face-net-config.npy
        weight_file_path = SiameseFaceNet.get_weight_path(model_dir_path)  # 设置存储的权重的路径
        checkpoint = ModelCheckpoint(weight_file_path, period=1)  # ModelCheckpoint 保存训练过程中的最佳模型权重

        # database是参与训练的每张照片，包括照片的编码；names是参与训练的照片的人名
        # t_y是一维矩阵，矩阵有24个数，t_y的shape为(24,)；t_x的shape为(24, 2, 1, 1000)
        t_x, t_y = self.create_pairs(database, names)

        print('data set pairs: ', t_x.shape)

        # print("model")
        # self.model.summary()
        # X[:,0]就是取所有行的第0个数据；X[:,1]就是取所有行的第1个数据
        hist = self.model.fit([t_x[:, 0], t_x[:, 1]], t_y,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.3,
                              verbose=SiameseFaceNet.VERBOSE,
                              callbacks=[checkpoint],
                              shuffle=True)
        # 将训练过程中的损失与精确度保存
        with open('loss_and_acc.txt', 'w') as f:
            f.write(str(hist.history))

        self.training_vis(hist)

        self.model.save_weights(weight_file_path)

    def verify(self, image_path, identity, database, threshold=None):
        self.threshold=0.6
        if threshold is not None:
            self.threshold = 0.6

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        # 计算图像的编码。 使用img_to_encoding（）参见上面的例子。计算的是要对比的一张图片的编码
        encoding = self.img_to_encoding(image_path,self.model)

        # Step 2: Compute distance with identity's image (≈ 1 line)
        '''
        # 用身份的图像计算距离，identity是提前存在库里的图片,即要对比的另一张图片
        input_pairs = []  # input_pairs是list类型,存的是两张图片的数字表达形式
        x = database[identity]  # x是list，里面很多数，数据的类型为float32
        for i in range(len(x)):
            input_pairs.append([encoding, x[i]])

        input_pairs = np.array(input_pairs)  # input_pairs类型为numpy.ndarray
        '''
        #dist = np.average(self.model.predict([input_pairs[:, 0], input_pairs[:, 1]]), axis=-1)[0]
        dist = float(np.linalg.norm(encoding - database[identity]))  # 求范数默认为2范数所有数的平方和开根号
        print(dist)

        # Step 3: Open the door if dist < threshold, else don't open (≈ 3 lines)
        # 如果dist <阈值则打开门，否则不要打开
        print(self.threshold)
        if dist < self.threshold:
            print("It's " + str(identity))
            is_valid = True
        else:
            print("It's not " + str(identity))
            is_valid = False

        return dist, is_valid



def main():
    # triplet_loss_test()
    model_dir_path = './models'
    image_dir_path = "./data/dlib-align-images"
    fnet = SiameseFaceNet()

    ''''#训练的代码
    database = dict()
    for filename in os.listdir(r"./data/dlib-align-images-test"):
        pic_list = []
        for picname in os.listdir(r"./data/dlib-align-images-test/" + filename):
            pic_list.append(fnet.img_to_encoding(image_dir_path + "/" + filename + "/" + picname,model_dir_path))
        database[filename] = pic_list
        # 将字典保存到文件中
    #np.save('picture_encoding_face.npy', database)

    # 读取文件中的内容，将内容赋给字典
    #read_dictionary = np.load("picture_encoding_face.npy").item()
    #database = read_dictionary

    # database是人名；
    fnet.fit(database=database, model_dir_path=model_dir_path)
    '''

    '''
    #训练的代码
    database = dict()
    for filename in os.listdir(r"./data/dlib-align-images"):
        pic_list = []
        for picname in os.listdir(r"./data/dlib-align-images/" + filename):
            pic_list.append(fnet.img_to_encoding(image_dir_path + "/" + filename + "/" + picname,model_dir_path))
        database[filename] = pic_list
        # 将字典保存到文件中
    np.save('picture_encoding_face.npy', database)
    
    # 读取文件中的内容，将内容赋给字典
    read_dictionary = np.load("picture_encoding_face.npy").item()
    database = read_dictionary

    # database是人名；
    fnet.fit(database=database, model_dir_path=model_dir_path)
    '''

    #'''#测试的代码
    model_dir_path = './models'
    # image_dir_path = "./data/images"
    image_dir_path = "./data/test/align"

    fnet = SiameseFaceNet()
    fnet.load_model(model_dir_path)

    database = {}
    database["danielle"] = fnet.img_to_encoding(image_dir_path + "/yanhan4.jpg",model_dir_path)
    fnet.verify(image_dir_path + "/yanhan5.jpg", "danielle", database)
    #'''


if __name__ == '__main__':
    main()
