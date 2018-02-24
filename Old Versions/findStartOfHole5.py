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
template0 = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)

cv2.imwrite('lFlapTemplate.jpg', template0)

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

_, last_frame = vid.read()
lastMaxVal = 1

percentDone = 0

print('Doing frame-by-frame analysis. Please wait...')

while vid.get(1) < vid.get(7):
    if vid.get(1)/vid.get(7)*100-percentDone >= 1:
        print(str(round(vid.get(1)/vid.get(7)*100,0))+'% done')
        percentDone+=1
    
    _, frame = vid.read()
    
    img = cv2.subtract(frame,last_frame)
    img = img[minXOR1:maxXOR1,minXOR2:maxXOR2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    tempMatch = cv2.matchTemplate(img, template0, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    if maxVal > .65:
        cv2.imwrite('zFound'+str(int(vid.get(1)))+'_1 (confidence='+str(round(maxVal,3))+'.jpg', last_frame)
        cv2.imwrite('zFound'+str(int(vid.get(1)))+'_2 (confidence='+str(round(maxVal,3))+'.jpg', frame)
    elif maxVal > .5:
        cv2.imwrite('zReject'+str(int(vid.get(1)))+'_1 (confidence='+str(round(maxVal,3))+'.jpg', last_frame)
        cv2.imwrite('zReject'+str(int(vid.get(1)))+'_2 (confidence='+str(round(maxVal,3))+'.jpg', frame)
    last_frame = frame