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
template3 = cv2.imread('0_template_small_1.png')
#template0 = cv2.cvtColor(template0, cv2.COLOR_BGR2GRAY)
template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3 = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)

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

vid.set(1,33550)

_, last_frame = vid.read()
maxVal1 = 0
last_maxVal = 0
percentDone = 47

print('Doing frame-by-frame analysis. Please wait...')
start_time = time.time()
while vid.get(1) < 36200-1:
    if vid.get(1)/vid.get(7)*100-percentDone >= 1:
        print(str(round(vid.get(1)/vid.get(7)*100,0))+'% done')
        percentDone+=1
    _, frame = vid.read()
    
    #cv2.imwrite('zref'+str(int(vid.get(1)))+'.jpg', frame)
    
    tl_corner = frame[0:100,int(vid.get(3)/4):int(3*vid.get(3)/4)]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
    tempMatch = cv2.matchTemplate(tl_corner, template2, cv2.TM_CCORR_NORMED, template2)
    minVal, maxVal1, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)

    if maxVal1 > 0.7 and not(hole_started):
        #print('hole began on frame ' + str(int(vid.get(1))) + ' (confidence:' + str(maxVal1)+')')
        #cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_1.jpg',last_frame)
        #cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_2.jpg',frame)
        hole_started = True
        for x in range(30):
            last_frame = frame
            _,frame = vid.read()
            op1 = cv2.bitwise_xor(frame,last_frame)
            op2 = cv2.bitwise_and(frame,last_frame)
            op = cv2.subtract(op1, op2)
            cv2.imshow('frame',frame)
            cv2.imshow('op',op)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
    elif maxVal1 > 0.7 and hole_started:
        pass
        #print('    hole is still in progress (frame '+ str(int(vid.get(1))) + ', confidence:' + str(maxVal1)+')')
    else:
        hole_started = False
        #print('    hole is over (frame ' + str(int(vid.get(1))) + ', confidence:' + str(maxVal1)+')')
        
    
    last_frame = frame
print('Job complete\nElapsed time: '+str(time.time()-start_time)+' seconds')