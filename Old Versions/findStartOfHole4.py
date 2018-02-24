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
int(vid.get(1)) - current frame (float)
vid.get(3) - vid width
vid.get(4) - vid height
vid.get(7) - total frames (float)

cv2.imwrite('filename', image)
ret, frame = vid.read()
'''
import cv2
import numpy as np

template0 = cv2.cvtColor(cv2.imread('lFlapTemplate.jpg'),cv2.COLOR_BGR2GRAY)

filename = ('test_big.mp4')

START = 1750
END = 4150

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
    maxXOR2 = int(vid.get(3))
else:
    raise Exception('video format ' + filename[-3:]+' not supported')

vid.set(1,START-1)

_, prev_frame = vid.read()
frames_processed = int(vid.get(1))

while frames_processed < END:
    vid.set(1,int(vid.get(1))+9)
    _, frame = vid.read()
    
    vidDiff = cv2.subtract(frame, prev_frame)
    vidDiff = vidDiff[minXOR1:maxXOR1,minXOR2:maxXOR2]
    vidDiff = cv2.cvtColor(vidDiff, cv2.COLOR_BGR2GRAY)
    
    tempMatch = cv2.matchTemplate(vidDiff, template0, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    
    if (maxVal < 0.7):
        print('    Skipped ' + str(int(vid.get(1)))+': max: '+str(maxVal))
        if maxVal>.2:
            cv2.imwrite('zskip'+str(int(vid.get(1)))+'_img1.jpg', prev_frame)
            cv2.imwrite('zskip'+str(int(vid.get(1)))+'_img2.jpg', frame)
            cv2.imwrite('zskip'+str(int(vid.get(1)))+'_diff.jpg', vidDiff)
        frames_processed += 10
    else:
        vid.set(1,frames_processed-1)
        _, prev_frame = vid.read()
        
        while (max < 0.7):
            _, frame = vid.read()
            
            vidDiff = cv2.subtract(frame, prev_frame)
            vidDiff = vidDiff[minXOR1:maxXOR1,minXOR2:maxXOR2]
            vidDiff = cv2.cvtColor(vidDiff, cv2.COLOR_BGR2GRAY)
            
            tempMatch = cv2.matchTemplate(vidDiff, template0, method=cv2.TM_CCORR_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
            
            prev_frame = frame
        print('    Target at frame ' + str(int(vid.get(1)))+': max: '+str(maxVal))
        cv2.imwrite('zfound'+str(int(vid.get(1)))+'_img.jpg', frame)
        cv2.imwrite('zfound'+str(int(vid.get(1)))+'_diff.jpg', vidDiff)
        
        frames_processed = int(vid.get(1))
    
    prev_frame = frame