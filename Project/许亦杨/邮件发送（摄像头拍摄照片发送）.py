# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:17:38 2020

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

#实现拍照：
def getPicture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite(path+'/person.jpg', frame)
    # 关闭摄像头
    cap.release()
    
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
    sendimagefile = open(path+'/person.jpg', 'rb').read()
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
    
msg = setMsg()
sendEmail(msg)

#判断网络连通状态：
'''
    判断网络是否联通,成功返回0，不成功返回1
    linux中ping命令不会自动停止，需要加入参数 -c 4，表示在发送指定数目的包后停止。
'''
def isLink():
    return os.system('ping -c 4 https://www.qq.com/?fromdefault')
    # return os.system('ping www.baidu.com')         
    
#主函数逻辑
'''
    如果网络连接正常，则拍照发邮件。 
    如果网络未连接，等待十秒钟再次测试，如果等待次数超过设置的最大次数，程序退出。
'''
def main():
    reconnect_times = 0
    while isLink():
        time.sleep(10)
        reconnect_times += 1
        if reconnect_times == exit_count:
            sys.exit(0)

    getPicture()
    msg = setMsg()
    sendEmail(msg)