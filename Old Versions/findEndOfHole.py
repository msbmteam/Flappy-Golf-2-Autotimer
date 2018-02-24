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

template1 = cv2.imread('tr_corner_template.png')
template2 = cv2.imread('holes_template.png')
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)

filename = 'test_big.mp4'
vid = cv2.VideoCapture(filename)
retryDetected = False
retry_visible = True

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

vid.set(1,33350)

_, last_frame = vid.read()

tr_corner = last_frame[0:int(vid.get(4)/2),int(vid.get(3)*3/4):int(vid.get(3))]
tr_corner = cv2.cvtColor(tr_corner,cv2.COLOR_BGR2GRAY)
tr_corner = cv2.adaptiveThreshold(tr_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
tempMatch = cv2.matchTemplate(tr_corner, template2, cv2.TM_CCORR_NORMED, template2)
minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)

while vid.get(1) < vid.get(7):
    _, frame = vid.read()
    
    tl_corner = frame[30:int(vid.get(4)/3),0:int(vid.get(3)/4)]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
    tempMatch = cv2.matchTemplate(tl_corner, template2, cv2.TM_CCORR_NORMED, template2)
    minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    
    
    if (maxVal1 > 0.9):
        print('menu appeared at frame ' + str(int(vid.get(1))) + ' (confidence:' + str(round(maxVal1,2))+')')
        cv2.imwrite('za_corner'+str(int(vid.get(1)))+'_1.jpg',last_frame)
        cv2.imwrite('za_corner'+str(int(vid.get(1)))+'_2.jpg',frame)
        placeholder = vid.get(1)
        vid.set(1,vid.get(1)-int(vid.get(5)/2))
        _,frame = vid.read()
        tr_corner = frame[0:100,int(vid.get(3)/4):int(vid.get(3))]
        tr_corner = cv2.cvtColor(tr_corner,cv2.COLOR_BGR2GRAY)
        tr_corner = cv2.adaptiveThreshold(tr_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
        tempMatch = cv2.matchTemplate(tr_corner, template1, cv2.TM_CCORR_NORMED, template1)
        minVal, last_maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
        maxVal = last_maxVal
        #print('quotient:'+ str(maxVal/last_maxVal))
        while maxVal/last_maxVal <= 1.02:
            #print('quotient:'+str(maxVal/last_maxVal))
            last_maxVal = maxVal
            vid.set(1,vid.get(1)-2)
            _,frame = vid.read()
            
            cv2.imshow('frame',frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            tr_corner = frame[0:100,int(vid.get(3)/4):int(vid.get(3))]
            tr_corner = cv2.cvtColor(tr_corner,cv2.COLOR_BGR2GRAY)
            tr_corner = cv2.adaptiveThreshold(tr_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
            tempMatch = cv2.matchTemplate(tr_corner, template1, cv2.TM_CCORR_NORMED, template1)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
            last_frame = frame
        print('frame of hole end:'+str(int(vid.get(1)+2)) + ', quotient ='+str(maxVal/last_maxVal))
        #print('quotient:'+str(maxVal/last_maxVal))
        cv2.imshow('frame',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        _,last_frame = vid.read()
        cv2.imwrite('za_corner'+str(int(vid.get(1)))+'_1.jpg',last_frame)
        _,frame = vid.read()
        cv2.imwrite('za_corner'+str(int(vid.get(1)))+'_2.jpg',frame)
        vid.set(1,int(round(placeholder+vid.get(5))))
        
    last_frame = frame