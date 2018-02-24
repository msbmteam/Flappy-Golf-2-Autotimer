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

import numpy as np
import cv2

template0 = cv2.imread('0-to-1-template.jpg')


vid = cv2.VideoCapture('SKZD3542.MOV')

_, prev_frame = vid.read()


while vid.get(1) < 50:
    printFrameNumber(vid)
    _, frame = vid.read()
    
    vidXOR = cv2.bitwise_xor(frame, prev_frame)
    result = cv2.matchTemplate(vidXOR, template0, method=cv2.TM_CCORR_NORMED)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    if (maxVal > 0.8):
        print('    min: '+str(minVal) + 'max: '+str(maxVal) + 'minLoc: '+str(minLoc)+'maxLoc: '+str(maxLoc))
        cv2.rectangle(frame, maxLoc, (maxLoc[0]+25, maxLoc[1]+25), (0,255,0),6)
        cv2.imwrite('found'+str(vid.get(1))+'.jpg', frame)
        cv2.imwrite('found'+str(vid.get(1))+'xor.jpg', vidXOR)
    
    
    prev_frame = frame
