# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:02:04 2020

@author: Administrator
"""


import io
import time
import picamera
import cv2
import numpy as np

# creat a stream
stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.start_preview()
    time.sleep(2)
    camera.capture(stream, format='jpeg')
# create numpy
data = np.fromstring(stream.getvalue(), dtype=np.uint8)
# recode from numpy to opencv
image = cv2.imdecode(data, 1)
# return rgb picture
image = image[:, :, ::-1]
cv2.imshow(image)
print('ok')
