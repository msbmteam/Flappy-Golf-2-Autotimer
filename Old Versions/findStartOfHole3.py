'''
Successfully parsed beginning frames of levels and frame that menu appears
Elapsed time: 30 seconds
Video length: 112 seconds
(Analysis speed: ~220 fps)

Untested on other stages, fps, resolutions
Unable to distinguish between beginning of level and beginning of menu
'''

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

template0 = cv2.imread('lFlapTemplate.jpg')

vid = cv2.VideoCapture('test_big.mp4')

_, prev_frame = vid.read()
frames_processed = vid.get(1)

while frames_processed < vid.get(7):
    vid.set(1,vid.get(1)+29)
    printFrameNumber(vid)
    _, frame = vid.read()
    
    vidXOR = cv2.bitwise_xor(frame, prev_frame)
    vidXOR = vidXOR[0:300,0:300]
    result = cv2.matchTemplate(vidXOR, template0, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    
    if (maxVal <= 0.8):
        frames_processed += 30
    else:
        vid.set(1,frames_processed-1)
        maxVal = 0
        while (maxVal <= 0.8):
            _, frame = vid.read()
            vidXOR = cv2.bitwise_xor(frame, prev_frame)
            vidXOR = vidXOR[0:300,0:300]
            result = cv2.matchTemplate(vidXOR, template0, method=cv2.TM_CCORR_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            prev_frame = frame
        print('    max: '+str(maxVal) +', maxLoc: '+str(maxLoc))
        cv2.rectangle(frame, maxLoc, (maxLoc[0]+25, maxLoc[1]+25), (0,255,0),6)
        cv2.imwrite('zfound'+str(vid.get(1))+'.jpg', frame)
        cv2.imwrite('zfound'+str(vid.get(1))+'xor.jpg', vidXOR)
        
        
        
        frames_processed = vid.get(1)
    
    prev_frame = frame