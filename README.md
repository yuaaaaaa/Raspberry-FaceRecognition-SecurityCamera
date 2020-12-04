# -树莓派智能安防摄像头（Smart security camera for home use by Raspberry-Pi）
![树莓派](README_files/2.jpg)
## -目录（Catalogue）
![目录](README_files/3.jpg)
## -描述（Description）
		RPi-AI-Camera是一款智能AI摄像头，具有人形侦查预警器。
		通过识别移动的物体，抓取人脸图像对比家庭成员信息来判别出外人入侵的情况，从而做到邮件预警的功能
## -详细功能（detailed function）
 * 移动物体的追踪定位框选
 * 通过人脸识别技术，对非家庭成员入侵邮件预警预警
 * 实时录像以减轻内存负担
  
  
  **GitHub仓库：**  [yuaaaaaa/RPi-AI-CAMERA](https://github.com/yuaaaaaa/RPi-AI-CAMERA.git)
  
## -为什么使用树莓派（Why Raspberry Pi?）
树莓派是一个只有信用卡大小的微型电脑，可以完成各种任务，基于他体型小的特点，可以用于家用微型摄像仪的载体。
这款Linux系统的计算机可以在低耗的情况下完成各种预期功能。
[了解更多](https://baike.so.com/doc/6240059-6453436.html)
## - 如何使用（How to do）
### step1-准备工具（tool）
* **硬件设备**
 * 树莓派（Raspberry-Pi）
 * 树莓派搭载摄像头（Raspberry Pi camera module）
* **软件设备**
 * Python3
 * openCV

### step2-配置Raspberry Pi（deploy）
* 确认openCV模块安装
```
进入Raspberry Pi命令界面
进入Python编译器 
>> python
测试opencv模块是否安装成功
>> import cv2
```
安装成功就可以进入下一步啦，如果出现了错误[点击这里](https://blog.csdn.net/kyokozan/article/details/79192646)
* 确认camera模块启动
```
// 尝试以下命令
ls -al /dev/ | grep video
// 当看到'video0 device'时表明摄像头模块已启动
```

### step3-测试相机（To test the camera）
#### 1) 现在让我们用树莓派拍一张照片吧~
```
// 拍摄一张照片 并命名为image.jpg存到本地
raspistill -o image.jpg
```
#### 2) 现在让我们用一段简单的代码测试一下吧
首先打开你的树莓派编译器，然后输入以下代码
```py
import cv2
cap = cv2.VideoCapture(0)
// 设置窗口的宽和高
cap.set(3,640)
cap.set(4,480)
while(True):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)
        k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
```
执行以上代码以后会出现两个运行窗口，frame和gray
在这里我们添加gray窗口是为了后续便于方便计算
![效果示意图](README_files/1.jpg)
### step4-移动物体目标检测（Moving object detection）
[代码仓库地址](https://github.com/yuaaaaaa/RPi-AI-CAMERA/blob/main/Project/%E9%99%88%E9%9B%A8%E6%99%B4/%E8%BF%90%E5%8A%A8%E7%89%A9%E4%BD%93%E6%A3%80%E6%B5%8B%E6%A1%86%E9%80%89.ipynb)
```python
"""
Created on Wed Nov 25 14:21:53 2020

@author: Administrator
"""

import cv2
import numpy as np
import os
path = "C://Users//Administrator//Desktop//项目实训//face_image"  # 模型数据图片目录
cap = cv2.VideoCapture(0)
total_image_name = []
total_face_encoding = []
for fn in os.listdir(path):  #fn 表示的是文件名q
    print(path + "/" + fn)
    fn = fn[:(len(fn) - 4)]  #截取图片名（这里应该把images文件中的图片名命名为为人物名）
    total_image_name.append(fn)  #图片名字列表

#读取视频流
cap = cv2.VideoCapture(0)
firstFrame = None
countflag = 0

if cap.isOpened():
    while(True):
        print('camera open')
        f1 = open('test.txt','w')
        f1.write('hello boy!')
        countflag += 1
        ret, frame = cap.read()
        #没有抓到第一帧那么说明到了视频结尾
        if not ret:
            break
        #调整帧的大小，转换为灰度图像进行高斯模糊
        framev = cv2.resize(frame, (640, 360))
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        #如果第一帧是None， 对其初始化
        if firstFrame is None:
            firstFrame = gray
            continue
        #计算当前帧与第一帧的不同
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        #扩展阀值图像填充孔洞
        thresh = cv2.dilate(thresh, None, iterations=3)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #遍历轮廓
        flag = 1
        for contour in contours:
            if cv2.contourArea(contour) < 1000: #面积阈值
                continue
            #计算最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if(flag):
                flag = 0
            
        cv2.putText(frame, "F{}".format(frame), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('frame_with_result', frame)
        cv2.imshow('thresh', thresh)
        cv2.imshow('frameDiff', frameDelta)
        if(countflag == 50):
            countflag = 0
            firstFrame = gray
        #处理按键效果
        key = cv2.waitKey(60) & 0xff
        if key == 27:
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        
    cap.release()
cv2.destroyAllWindows()
```
**tips：**
 * 高斯模糊

高斯模糊是一种广泛使用的图形软件的方法，通常会减少图像噪声和减少细节,适用于很多的场合和情景；
高斯模糊的核心就是取中心点周围所有像素点的均值作为自己的像素值，以此来达到平滑效果；
在本例中使用高斯模糊对的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。 

 * 二值化阈值处理

二值化阈值处理就是将图像上的像素点的灰度值设置为0或255，也就是将整个图像呈现出明显的只有黑和白的视觉效果。
在本例中对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map），
还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，
从而对孔（hole）和缺陷（imperfection）进行归一化处理。

### step5-人脸检测（Face Detection）
[代码仓库地址](https://github.com/yuaaaaaa/RPi-AI-CAMERA/blob/main/Project/%E9%99%88%E9%9B%A8%E6%99%B4/%E5%9B%BE%E5%83%8F%E4%BA%BA%E8%84%B8%E6%A1%86%E9%80%89.ipynb)
```python
import cv2

def discern(img): # 图片识别方法封装
    pathf = '/Users/chenyuqing/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(pathf)
    faceRects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    cv2.imshow("Image", img)


cap = cv2.VideoCapture(0)  # 获取摄像头0表示第一个摄像头
while (1):  # 逐帧显示
    ret, img = cap.read()
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源

```
**tips:**
* 人脸检测

人脸识别最基本的任务当然是“人脸检测”。与将来拍摄的新面孔相比，您必须先“捕捉”面孔以识别它。
最常用的当然是Haar（哈尔）分类器，Haar（哈尔）分类器：Haar(哈尔)特征分为以下几类类：边缘特征、线性特征、中心特征和对角线特征，组合成特征模板。
特别的，OpenCV自带了训练器和检测器（Cascade Classifier Training），该XML文件，就是OpenCV自带的检测器。

### step6-人脸识别（Face Recognition）
[数据集](https://github.com/polarisZhao/awesome-face)

[代码仓库地址](https://github.com/yuaaaaaa/RPi-AI-CAMERA/blob/main/Project/%E9%99%88%E9%9B%A8%E6%99%B4/%E5%9B%BE%E5%83%8F%E4%BA%BA%E8%84%B8%E6%A3%80%E6%B5%8B.ipynb)
```python
#人脸识别类 - 使用face_recognition模块
import cv2
import face_recognition
import os

path = "/Users/chenyuqing/Desktop/face_image "  # 模型数据图片目录
cap = cv2.VideoCapture(0)
total_image_name = []
total_face_encoding = []
for fn in os.listdir(path):  #fn 表示的是文件名q
    print(path + "/" + fn)
    total_face_encoding.append(
        face_recognition.face_encodings(
            face_recognition.load_image_file(path + "/" + fn))[0])
    fn = fn[:(len(fn) - 4)]  #截取图片名（这里应该把images文件中的图片名命名为为人物名）
    total_image_name.append(fn)  #图片名字列表
while (1):
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame) # 发现在视频帧所有的脸和face_enqcodings
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):     # 在这个视频帧中循环遍历每个人脸
        for i, v in enumerate(total_face_encoding):        # 看看面部是否与已知人脸相匹配。
            match = face_recognition.compare_faces(
                [v], face_encoding, tolerance=0.5)
            name = "Unknown"
            if match[0]:
                name = total_image_name[i]
                break

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)         # 画出一个框，框住脸
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
                      cv2.FILLED)        # 画出一个带名字的标签，放在框下
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                    (255, 255, 255), 1)
    cv2.imshow('Video', frame)     # 显示结果图像
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
**tips:**
* Face_Recognition

Face_Recognition人脸识别模型是一个具有29个转换层的ResNet(残差网络)。

### step7-定时录像（Regular mail）
[代码仓库地址](https://github.com/yuaaaaaa/RPi-AI-CAMERA/blob/main/Project/%E8%AE%B8%E4%BA%A6%E6%9D%A8/%E5%AE%9E%E7%8E%B0%E5%BD%95%E5%83%8F.ipynb)
```python
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

#创建VideoWrite对象，10是fps（每秒读取的帧数），(640,480)是屏幕大小
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./cameraoutput.avi', fourcc, 25, (640,480))

video_len = 250
if cap.isOpened():
    while(video_len>0):
        '''
            cv2.VideoCapture(0).read()
                功能：读取一帧的图片
                参数：无
                返回值：1.（boolean值）是否读取到图片
                        2.一帧图片
        '''
        ret, frame = cap.read()
        #保存写入这一帧图像frame:
        out.write(frame)
        #显示这一帧图像frame:
        cv2.imshow("capture", frame)
        #当按下'q'键时退出：
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_len -= 1


cap.release()
out.release()
cv2.destroyAllWindows()
```
**tip:**
* 定时录像

当画面静止时间超过6s时，该摄像系统将不再录像记录，以达到节省本地内存的优点

### step8-非家庭成员入侵发邮件报警（Stanger call the police）
[代码仓库地址](https://github.com/yuaaaaaa/RPi-AI-CAMERA/blob/main/Project/%E8%AE%B8%E4%BA%A6%E6%9D%A8/%E9%82%AE%E4%BB%B6%E5%8F%91%E9%80%81%EF%BC%88%E6%91%84%E5%83%8F%E5%A4%B4%E6%8B%8D%E6%91%84%E7%85%A7%E7%89%87%E5%8F%91%E9%80%81%EF%BC%89.ipynb)
```python
"""
Created on Sun Nov 29 17:16:14 2020

@author: 32991
"""

import cv2
import smtplib
import sys
import os
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

#设置参数：
smtpserver  = '3299115821@qq.com'     # smtp服务器
username    = '3299115821@qq.com'     # 发件邮箱账号
password    = 'ndudpbtzsczddcac'             # 邮箱登录密码
sender      = '3299115821@qq.com'     # 发件人
addressee   = '3299115821@qq.com'     # 收件人
exit_count  = 10                       # 尝试联网次数
path        = os.getcwd()             #获取图片保存路径 
'''
    os.getcwd() 方法
        用于返回当前工作目录，无参数
'''

#构造邮件内容：
def setMsg():     # 下面依次为邮件类型，主题，发件人和收件人。
    msg = MIMEMultipart('mixed') 
    msg['Subject'] = '出现非家庭成员！'
    msg['From'] = '3299115821@qq.com <3299115821@qq.com>'
    msg['To'] = addressee

    text = "主人，出现非家庭成员！照片如下！"     # 下面为邮件的正文
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)

    sendimagefile = open(path+'/person.jpg', 'rb').read()     # 构造图片链接
    image = MIMEImage(sendimagefile)
    image["Content-Disposition"] = 'attachment; filename="people.png"'     # 下面一句将收件人看到的附件照片名称改为people.png。
    msg.attach(image)
    return msg.as_string()

#实现邮件发送：
def sendEmail(msg):
    #发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.qq.com')
    smtp.login(username, password)
    smtp.sendmail(sender, addressee, msg)
    smtp.quit()
    
msg = setMsg()
sendEmail(msg)
```

**效果渲染**

```邮件接收```

![邮件接收](README_files/2.png)

```不属于家庭成员情况```

![不属于家庭成员情况](README_files/3.png)

```属于家庭成员情况```

![属于家庭成员情况](README_files/4.png)

```确认为家庭成员```

![确认为家庭成员](README_files/6.png)

### step9-CNN神经网络（CNNNET）
[代码仓库](https://github.com/yuaaaaaa/Raspberry-FaceRecognition-SecurityCamera/blob/main/Project/%E9%99%88%E9%9B%A8%E6%99%B4/model.py)

```python
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
        #本案例是基于thano的，如果基于tensorflow的backend需要进行修改
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
    
        
```
**tips:**
![卷积神经网络](README_files/1.png)
* 基于卷积神经网络完成人脸识别

## -性能描述
### 1）关于移动物体识别
对于每一帧 计算得到： 
每处理一帧像素并进行计算 计算机 需要使用```31.26933333333333```毫秒
每处理一帧像素并进行计算 树莓派 需要使用```93.80799999999999```毫秒

### 2）对于人脸识别
对于每一帧 计算得到： 

每处理一帧像素并进行计算 笔记本 需要使用```49.980711323233333```毫秒

每处理一帧像素并进行计算 树莓派 需要使用```149.94213487503316```毫秒

### 3）对于邮件发送
对于一封邮件 计算得到：

每发送一封邮件 笔记本 需要使用```1564.6496629999547```毫秒

每发送一封邮件 树莓派 需要使用```1428.1979999999996```毫秒

### 4）对于定时录像
对于每一次启动该程序

每完整的启动一次该程序 计算机 需要```3732.4840199999016```毫秒

每完整的启动一次该程序 树莓派 需要```22.181```毫秒