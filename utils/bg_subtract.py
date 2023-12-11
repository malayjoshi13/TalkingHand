import numpy as np
import cv2

def removeBG(frame, bgModel, learningRate):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res