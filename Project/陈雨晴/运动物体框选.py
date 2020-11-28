import cv2
 
vc = cv2.VideoCapture("/Users/chenyuqing/Downloads/try.MP4")  # 读入视频文件
# vc = cv2.VideoCapture("C:/Users/jason/Desktop/152821AA.MP4")
 
rval, firstFrame = vc.read()
firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_CUBIC)
gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)   # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)      #高斯模糊，用于去噪
prveFrame = firstFrame.copy()
 
 
#遍历视频的每一帧
while True:
    (ret, frame) = vc.read()
 
    # 如果没有获取到数据，则结束循环
    if not ret:
        break
 
    # 对获取到的数据进行预处理
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    cv2.imshow("current_frame", gray_frame)
    cv2.imshow("prveFrame", prveFrame)
 
    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(prveFrame, gray_frame)
    cv2.imshow("frameDiff", frameDiff)
    prveFrame = gray_frame.copy()
 
 
    # 忽略较小的差别
    retVal, thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)
 
 
    # 对阈值图像进行填充补洞
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    text = "Unoccupied"
    # 遍历轮廓
    for contour in contours:
        # if contour is too small, just ignore it
        if cv2.contourArea(contour) < 50:   #面积阈值
            continue
 
        # 计算最小外接矩形（非旋转）
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied!"
 
    # cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "F{}".format(frame), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
    cv2.imshow('frame_with_result', frame)
    cv2.imshow('thresh', thresh)
    cv2.imshow('frameDiff', frameDiff)
 
    # 处理按键效果
    key = cv2.waitKey(60) & 0xff
    if key == 27:  # 按下ESC时，退出
        break
    elif key == ord(' '):  # 按下空格键时，暂停
        cv2.waitKey(0)
 
    cv2.waitKey(0)
vc.release()