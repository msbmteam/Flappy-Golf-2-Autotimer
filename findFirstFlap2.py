def printFrameNumber(vid):
    print('frame ' + str(int(vid.get(1))) + ' of ' + str(int(vid.get(7))))

'''
vid.get(1) - current frame (float)
vid.get(3) - vid width
vid.get(4) - vid height
vid.get(7) - total frames (float)

cv2.imwrite('filename', image)
ret, frame = vid.read()
'''
import cv2
import numpy as np
import time

#template0 = cv2.imread('lFlapTemplate.jpg')
template1 = cv2.imread('tr_corner_template.png')
template2 = cv2.imread('star_template.png')
template3 = cv2.imread('1_small_template.png')
template4 = cv2.imread('1_small_template2.png')
#template0 = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3 = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
template4 = cv2.cvtColor(template4, cv2.COLOR_BGR2GRAY)

filename = 'test_big.mp4'
vid = cv2.VideoCapture(filename)
hole_started = False

if filename[-3:].lower() == 'mov':
    minXOR1 = 0
    maxXOR1 = 300
    minXOR2 = 0
    maxXOR2 = 300
elif filename[-3:].lower() == 'mp4':
    minXOR1 = int(vid.get(4))-300
    maxXOR1 = int(vid.get(4))
    minXOR2 = 0
    maxXOR2 = 300
else:
    raise Exception('video format ' + filename[-3:]+' not supported')

vid.set(1,33650)
last_maxVal = 0
_, last_frame = vid.read()
print('Doing frame-by-frame analysis. Please wait...')
start_time = time.time()
while vid.get(1) < 36200-1:
    _, frame = vid.read()
    
    frametmp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frametmp = cv2.adaptiveThreshold(frametmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-10)
    tempMatch = cv2.matchTemplate(frametmp, template3, cv2.TM_CCORR_NORMED, template3)
    tempMatchIMG = cv2.normalize(tempMatch, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    _,maxVal,_,location0 = cv2.minMaxLoc(tempMatch)
    
    if maxVal-last_maxVal > 0.05 or maxVal > 0.7:
        print(maxVal)
        cv2.rectangle(frame, location0, (location0[0]+25, location0[1]+25), (0,0,255),1)
        cv2.imshow('frame',frame)
        cv2.imshow('tempMatch1',tempMatch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    last_maxVal = maxVal
    last_frame = frame