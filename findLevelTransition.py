def printFrameNumber(vid):
    print('frame ' + str(int(vid.get(1))) + ' of ' + str(int(vid.get(7))))

'''
vid.get(1) - current frame (float)
vid.get(3) - vid width
vid.get(4) - vid height
vid.get(5) - vid FPS
vid.get(7) - total frames (float)

cv2.imwrite('filename', image)
ret, frame = vid.read()
'''
import cv2
import numpy as np

#template0 = cv2.imread('lFlapTemplate.jpg')
template1 = cv2.imread('tr_corner_template.png')
template2 = cv2.imread('star_template.png')
#template0 = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)

filename = 'test_big.mp4'
vid = cv2.VideoCapture(filename)

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

vid.set(1,1500)

_, prev_frame = vid.read()
tl_corner = prev_frame[0:50,0:50]
tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
prev_minVal, prev_maxVal, minLoc, maxLoc = cv2.minMaxLoc(tl_corner)
while vid.get(1) < vid.get(7):
    _, frame = vid.read()
    
    tl_corner = frame[0:50,0:50]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tl_corner)
    
    if abs(maxVal-minVal) < 3 and maxVal < 10 and not(abs(prev_maxVal-prev_minVal) < 3 and prev_maxVal < 10):
        print('    transition detected (difference='+str(prev_maxVal-maxVal))
        cv2.imwrite('transition'+str(int(vid.get(1)))+'_1.jpg', prev_frame)
        cv2.imwrite('transition'+str(int(vid.get(1)))+'_2.jpg', frame)
    
    prev_frame = frame
    prev_maxVal = maxVal
    prev_minVal = minVal