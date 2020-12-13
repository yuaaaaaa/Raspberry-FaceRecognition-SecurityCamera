# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:51:39 2020

@author: Administrator
"""


# arr_list = [53.29769100001158,49.432657000011204,49.60332300004211,50.925989000006666,48.899148000032255,49.64951599993128,48.15971000004993,49.877659000003405]
# print((sum(arr_list)/8))

arr_list = [93.97,93.187,94.042,95.026,93.782,94.146,95.194,94.127,93.269,93.293,91.852]
print(sum(arr_list)/11)


# import cv2
# import time

# # 图片识别方法封装
# def discern(img):
#     pathf = 'C:\\Users\\Administrator\\Desktop\\project practice\\haarcascades\\haarcascade_frontalface_default.xml'
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(pathf)
#     faceRects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
#     if len(faceRects):
#         for faceRect in faceRects:
#             x, y, w, h = faceRect
#             cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
#     cv2.imshow("Image", img)


# # 获取摄像头0表示第一个摄像头
# cap = cv2.VideoCapture(0)

# while (1):  # 逐帧显示
#     time_start = time.clock()
#     ret, img = cap.read()
#     # cv2.imshow("Image", img)
#     discern(img)
#     time_end = time.clock()
#     print((time_end - time_start)*1000)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()  # 释放摄像头
# cv2.destroyAllWindows()  # 释放窗口资源
