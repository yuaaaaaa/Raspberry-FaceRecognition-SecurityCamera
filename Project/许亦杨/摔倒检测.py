# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:27:52 2020

@author: 32991
"""

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
import poplib,email,telnetlib
import datetime,time,sys,traceback
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr
from PIL import Image, ImageDraw, ImageFont

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


'''
    接收返回邮件
'''
class down_email():

    def __init__(self,user,password,eamil_server):
        # 输入邮件地址, 口令和POP3服务器地址:
        self.user = user
        # 此处密码是授权码,用于登录第三方邮件客户端
        self.password = password
        self.pop3_server = eamil_server

    # 获得msg的编码
    def guess_charset(self,msg):
        charset = msg.get_charset()
        if charset is None:
            content_type = msg.get('Content-Type', '').lower()
            pos = content_type.find('charset=')
            if pos >= 0:
                charset = content_type[pos + 8:].strip()
        return charset

    #获取邮件内容
    def get_content(self,msg):
        content=''
        content_type = msg.get_content_type()
        # print('content_type:',content_type)
        if content_type == 'text/plain': # or content_type == 'text/html'
            content = msg.get_payload(decode=True)
            charset = self.guess_charset(msg)
            if charset:
                content = content.decode(charset)
        return content

    # 字符编码转换
    # @staticmethod
    def decode_str(self,str_in):
        value, charset = decode_header(str_in)[0]
        if charset:
            value = value.decode(charset)
        return value

    # 解析邮件,获取附件
    def get_att(self,msg_in, str_day, Subject):
        attachment_files = []
        for part in msg_in.walk():
            # 获取附件名称类型
            file_name = part.get_param("name")  # 如果是附件，这里就会取出附件的文件名
            # file_name = part.get_filename() #获取file_name的第2中方法
            # contType = part.get_content_type()
            if file_name:
                h = email.header.Header(file_name)
                # 对附件名称进行解码
                dh = email.header.decode_header(h)
                filename = dh[0][0]
                if dh[0][1]:
                    # 将附件名称可读化
                    filename = self.decode_str(str(filename, dh[0][1]))
                    # print(filename)
                    # filename = filename.encode("utf-8")
                if Subject == '该人员属于家庭成员':
                    # 下载附件
                    data = part.get_payload(decode=True)
                    # 在指定目录下创建文件，注意二进制文件需要用wb模式打开
                    att_file = open('./img/face_recognition/' + filename, 'wb')
                    att_file.write(data)  # 保存附件
                    att_file.close()
                    attachment_files.append(filename)
        return attachment_files

    def run_ing(self):
        str_day = str(datetime.date.today())# 日期赋值
        # 连接到POP3服务器,有些邮箱服务器需要ssl加密，可以使用poplib.POP3_SSL
        try:
            telnetlib.Telnet(self.pop3_server, 995)
            self.server = poplib.POP3_SSL(self.pop3_server, 995, timeout=10)
        except:
            time.sleep(5)
            self.server = poplib.POP3(self.pop3_server, 110, timeout=10)

        # server.set_debuglevel(1) # 可以打开或关闭调试信息
        # 打印POP3服务器的欢迎文字:
        #print(self.server.getwelcome().decode('utf-8'))
        # 身份认证:
        self.server.user(self.user)
        self.server.pass_(self.password)
        # 返回邮件数量和占用空间:
        #print('Messages: %s. Size: %s' % self.server.stat())
        # list()返回所有邮件的编号:
        resp, mails, octets = self.server.list()
        # 可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
        #print(mails)
        index = len(mails)
        for i in range(index, 0, -1):# 倒序遍历邮件
        # for i in range(1, index + 1):# 顺序遍历邮件
            resp, lines, octets = self.server.retr(i)
            # lines存储了邮件的原始文本的每一行,
            # 邮件的原始文本:
            msg_content = b'\r\n'.join(lines).decode('utf-8')
            # 解析邮件:
            msg = Parser().parsestr(msg_content)
            #获取邮件的发件人，收件人， 抄送人,主题
            # hdr, addr = parseaddr(msg.get('From'))
            # From = self.decode_str(hdr)
            # hdr, addr = parseaddr(msg.get('To'))
            # To = self.decode_str(hdr)
            # 方法2：from or Form均可
            From = parseaddr(msg.get('from'))[1]
            To = parseaddr(msg.get('To'))[1]
            Cc=parseaddr(msg.get_all('Cc'))[1]# 抄送人
            Subject = self.decode_str(msg.get('Subject'))
            #print('from:%s,to:%s,Cc:%s,subject:%s'%(From,To,Cc,Subject))
            # 获取邮件时间,格式化收件时间
            date1 = time.strptime(msg.get("Date")[0:24], '%a, %d %b %Y %H:%M:%S')
            # 邮件时间格式转换
            date2 = time.strftime("%Y-%m-%d",date1)
            if date2 < str_day:
                break # 倒叙用break
                # continue # 顺叙用continue
            else:
                # 获取附件
                attach_file=self.get_att(msg,str_day,Subject)
                #print(attach_file)

        # 可以根据邮件索引号直接从服务器删除邮件:
        # self.server.dele(7)
        self.server.quit()

'''
    摔倒检测
'''
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
cap.set(3, 640)  # set video weight
cap.set(4, 480)  # set video height
scale = 0
# 背景减除,实例化混合高斯模型
fg = cv2.createBackgroundSubtractorMOG2()
        
if cap.isOpened():
    #创建VideoWrite对象，25是fps（每秒读取的帧数），(640,480)是屏幕大小
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./cameraoutput.avi', fourcc, 10, (640,480))
    send_time = 0
    confirm_time = 0
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
            
            
        #确认读取到该帧图像的时刻，收件箱中是否有确认信息
        if confirm_time <= 0:
            try:
                # 输入邮件地址, 口令和POP3服务器地址:
                user = '3299115821@qq.com'
                # 此处密码是授权码,用于登录第三方邮件客户端
                password = 'zfzmzhjuruuwciib'
                eamil_server = 'pop.qq.com'
                # user='xinfei@.com'
                # password = 'f67h2'
                # eamil_server = 'pop.exmail.qq.com'
                email_class=down_email(user=user,password=password,eamil_server=eamil_server)
                email_class.run_ing()
            except Exception as e:
                import traceback
                ex_msg = '{exception}'.format(exception=traceback.format_exc())
            confirm_time = 11   #设置和发送时间宽度一致
        confirm_time -= 1
        
    cap.release()
cv2.destroyAllWindows()
img