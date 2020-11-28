import os
import cv2
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random

import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
path = "/Users/chenyuqing/Desktop/dataset"
IMG_SIZE = 128
#建立一个用于存储和格式化读取训练数据的类
class DataSet(object):
    def __init__(self,path):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.img_size = 128
        self.extract_data(path) #在这个类初始化的过程中读取path下的训练数据

    def extract_data(self,path):
        #根据指定路径读取出图片、标签和类别数
        labels = []
        imgs = []
        names = []
        count = 0
        for fn in os.listdir(path):  #fn 表示的是文件名q
            if fn == '.DS_Store':
                continue
            print(path + "/" + fn)
            img = plt.imread(path + "/" + fn)
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
            imgs.append(np.array(recolored_img))
            fn = fn[:(len(fn) - 4)]  #截取图片名（这里应该把images文件中的图片名命名为为人物名）
            names.append(fn)  #图片名字列表
            labels.append(count)
            count += 1
        counter = count
        labels =np.array(labels)
        imgs = np.array(imgs)
        #print(imgs)
        #将数据集打乱随机分组
        X_train,X_test,y_train,y_test = train_test_split(imgs,labels,test_size=0.2,random_state=random.randint(0, 100))

        #重新格式化和标准化
        # 本案例是基于thano的，如果基于tensorflow的backend需要进行修改
        print(X_train.shape)
        X_train = X_train.reshape(X_train.shape[0], 1, self.img_size, self.img_size)/255.0
        X_test = X_test.reshape(X_test.shape[0], 1, self.img_size, self.img_size) / 255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        #将labels转成 binary class matrices
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)
    
        #将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter


#建立基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "/Users/chenyuqing/大三上/实训/model.h5" 
    def __init__(self):
        self.model = None
        
    #读取实例化后的Dataset类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset
        
    #建立CNN模型，一层卷积，一层池化， 一层卷积，一层池化，名之后进行全连接最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            ))
        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            ))
        
        
        self.model.add(
            Convolution2D(
                filters=64, 
                kernel_size=(5, 5),
                padding='same'
        ))
        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            ))
        
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()
    
    #进行模型训练，改变不同optimizer,loss
    def train_model(self):
        self.model.compile(
            optimizer='adam', #RMSprop,Adagrad
            loss='categorical_crossentropy', #squared_hinge
            metrics=['accuracy']
        )
        #epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=7, batch_size=20)
      
    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)
    
    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)
        
    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)
    
    #需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self,img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)  #测算一下该img属于某个label的概率
        max_index = np.argmax(result) #找出概率最高的

        return max_index,result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率

if __name__ == '__main__':
    dataset = DataSet('/Users/chenyuqing/Desktop/dataset')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
    
        