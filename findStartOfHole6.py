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

#template0 = cv2.imread('lFlapTemplate.jpg')
template1 = cv2.imread('tr_corner_template.png')
template2 = cv2.imread('star_template.png')
#template0 = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)

filename = 'test_big.mp4'
vid = cv2.VideoCapture(filename)

percentDone = 0

_, frame = vid.read()
tl_corner = frame[0:100,int(vid.get(3)/4):int(3*vid.get(3)/4)]
tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
tempMatch = cv2.matchTemplate(tl_corner, template2, method=cv2.TM_CCORR_NORMED)
minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)

while maxVal1 > 0.7:
    _, frame = vid.read()
    tl_corner = frame[0:100,int(vid.get(3)/4):int(3*vid.get(3)/4)]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
    tempMatch = cv2.matchTemplate(tl_corner, template2, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)

while vid.get(1) < vid.get(7):
    _, frame = vid.read()
    
    #cv2.imwrite('zref'+str(int(vid.get(1)))+'.jpg', frame)
    
    tl_corner = frame[0:100,int(vid.get(3)/4):int(3*vid.get(3)/4)]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
    tempMatch = cv2.matchTemplate(tl_corner, template2, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)

    if maxVal1 > 0.7:
        #print('hole began on frame ' + str(int(vid.get(1))) + ' (confidence:' + str(maxVal1)+')')
        cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_1.jpg',last_frame)
        cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_2.jpg',frame)
        cv2.imshow('star',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
    if vid.get(1)/vid.get(7)*100-percentDone >= 1:
        print(str(round(vid.get(1)/vid.get(7)*100,0))+'% done')
        percentDone+=1
    last_frame = frame