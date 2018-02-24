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

filename = 'numbers demo.mp4'
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

_, last_frame = vid.read()
maxVal1 = 0
last_maxVal = 0

print('Doing frame-by-frame analysis. Please wait...')

while vid.get(1) < 1850:
    _, frame = vid.read()
    
    cv2.imwrite('zref'+str(int(vid.get(1)))+'.png', frame)