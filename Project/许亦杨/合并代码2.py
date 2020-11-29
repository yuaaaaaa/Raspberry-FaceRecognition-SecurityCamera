# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:11:13 2020

@author: 32991
"""

'''
    定时发送：大概15秒一次
'''
import cv2
import numpy as np
import face_recognition
import os
import smtplib
import sys
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


#设置发送邮件功能参数：
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
def setMsg():
    # 下面依次为邮件类型，主题，发件人和收件人。
    msg = MIMEMultipart('mixed')
    msg['Subject'] = '出现非家庭成员！'
    msg['From'] = '3299115821@qq.com <3299115821@qq.com>'
    msg['To'] = addressee

    # 下面为邮件的正文
    text = "主人，出现非家庭成员！照片如下！"
    text_plain = MIMEText(text, 'plain', 'utf-8')
    msg.attach(text_plain)

    # 构造图片链接
    sendimagefile = open(path1+'/external_personnel.jpg', 'rb').read()
    image = MIMEImage(sendimagefile)
    # 下面一句将收件人看到的附件照片名称改为people.png。
    image["Content-Disposition"] = 'attachment; filename="people.png"'
    msg.attach(image)
    return msg.as_string()

#实现邮件发送：
def sendEmail(msg):
    # 发送邮件
    smtp = smtplib.SMTP()
    smtp.connect('smtp.qq.com')
    smtp.login(username, password)
    smtp.sendmail(sender, addressee, msg)
    smtp.quit()


path = './img/face_recognition'  # 模型数据图片目录
path1 = './img'                  # 检测Unknown人脸图像放置目录
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
#读取视频流
cap = cv2.VideoCapture(0)
firstFrame = None
countflag = 0
video_transcribe_time = 0

#设置视频参数
#cap.set(3, 480)
def discern(img,send_time):
    pathf = './haarcascades/haarcascade_frontalface_default.xml'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(pathf)
    faceRects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        #for faceRect in faceRects:
            #x, y, w, h = faceRect
            #cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        # 在这个视频帧中循环遍历每个人脸
        for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings):
            # 看看面部是否与已知人脸相匹配。
            for i, v in enumerate(total_face_encoding):
                match = face_recognition.compare_faces(
                    [v], face_encoding, tolerance=0.5)
                name = "Unknown"
                color = (0, 0, 255)
                if match[0]:
                    name = total_image_name[i]
                    color = (0, 255, 0)
                    break
            
            
            #判断为外部人员，则发送该人脸图像信息给主人：
            if (name == 'Unknown') and (send_time <= 0):
                cv2.imwrite(path1+'/external_personnel.jpg', frame)
                msg = setMsg()
                sendEmail(msg)
                send_time = 11
                
            # 画出一个框，框住脸
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # 画出一个带名字的标签，放在框下
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color,cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0,
                        (255, 255, 255), 1)
        # 显示结果图像
        cv2.imshow('Video', frame)
    send_time -= 1
    return send_time


if cap.isOpened():
    #创建VideoWrite对象，25是fps（每秒读取的帧数），(640,480)是屏幕大小
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./cameraoutput.avi', fourcc, 10, (640,480))
    send_time = 0
    while(True):
        countflag += 1
        ret, frame = cap.read()
        #没有抓到第一帧那么说明到了视频结尾
        frame_write = frame
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
        
        '''
            如果发现存在运动物体则video_transcribe_time设置为250：
                frameDelta：返回与上一帧的不同数组
                frameDelta_shape:获取frameDelta的大小
                compare_array:构建全0数组进行对比
            
        '''
        frameDelta_shape = frameDelta.shape
        compare_array = np.zeros(frameDelta_shape)
        if (frameDelta != compare_array).all():
            video_transcribe_time = 60
        if video_transcribe_time>0:
            #保存写入这一帧图像frame:
            out.write(frame_write)
            video_transcribe_time -= 1
            
        #thresh为获得的阈值图：
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
                send_time = discern(framev,send_time)
                
            
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