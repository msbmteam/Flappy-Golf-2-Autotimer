'''
findEndOfHole2.py
Flappy Golf 2 video analysis script by msbmteam

This script saves to file all frames in which the between level menu appears and
the frames before and after entry into a hole.

These frames are of interest to us because"menu load times" start on the first
frame the player enters the hole and stop on the first frame that the between-
level menu appears.

Summary of script:
    1. Find the next frame the between-level menu appears. Save images to file
    2. Go backward frame-by-frame for 2 seconds to see which frame the player
         entered the hole. Save images to file
    3. Repeat steps 1 and 2 until the end of the video is reached

We find the first frame the between-level menu appears first for two reasons:
The holes button always looks the same regardless of stage, and it always appears
in the same location onscreen, so it's easy to detect it.  Also, after a player
enters the hole, the camera stops moving; this makes it so there is less noise
when XOR-ing the top-right corner to look for the frame when the pause/retry
buttons appear.

We have to analyze 2 seconds of frames to search for the frame the player entered
the hole because the pause/retry buttons are partially transparent, and confidence
values can swing drastically with different colored backgrounds. (White-background
levels like Line City are especially problematic.) By iterating through 2 seconds
of frames, we can find a LOCAL max confidence value for the frame the menu appears

Eventually this code will be incorporated into a big "time run" or "remove loads"
script. (Frame numbers will be saved and load times will be subtracted out)

Currently this script can only support 720p mp4 files
'''

#Prints the current frame of VideoCapture object vid
def printFrameNumber(vid):
    print('frame ' + str(int(vid.get(1))) + ' of ' + str(int(vid.get(7))))

import cv2
import numpy as np
import time

#template1 = cv2.imread('tr_corner_template.png')
template2 = cv2.imread('holes_template.png')
template3 = cv2.imread('tr_corner_template2.png')
#template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3 = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)

filename = 'test_big.mp4'
print("Importing video file "+filename+"...")
vid = cv2.VideoCapture(filename)

# sets the frame count to START_FRAME. The first frame read will be START_FRAME+1
START_FRAME = -1
vid.set(1,START_FRAME)

# start the loop
END_FRAME = vid.get(7)-1
start_time = time.time()
print("Commencing analysis of "+str(int(END_FRAME-START_FRAME))+" frames...")
while vid.get(1) < END_FRAME:
    # See if the next frame has the between-level menu visible (specifically looks for holes button)
    _, frame = vid.read()
    tl_corner = frame[30:int(vid.get(4)/3),0:int(vid.get(3)/4)]
    tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
    tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
    tempMatch = cv2.matchTemplate(tl_corner, template2, method=cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    
    if (maxVal > 0.9): # We can use a high threshold because the button always looks the same
        filename2 = 'z_'+str(int(vid.get(1)))+'_menu2.jpg'
        cv2.imwrite(filename2,frame)
        placeholder = vid.get(1)+1
        vid.set(1,vid.get(1)-2)
        _,frame = vid.read()
        filename3 = 'z_'+str(int(vid.get(1)))+'_menu1.jpg'
        cv2.imwrite(filename3,frame)
        print('menu appeared at frame ' + str(int(vid.get(1))) + ' (confidence:'
              + str(round(maxVal,2))+')' + '. Saved frames to '+filename2+' and '+filename3)
        
        # Go backward frame-by-frame to find the frame the pause/retry buttons in 
        # the top-right corner appear onscreen
        vid.set(1,vid.get(1)-2)
        _,frame = vid.read()
        tr_corner = frame[0:100, int(vid.get(3)-150):int(vid.get(3))]
        maxVal = 0
        # Go backwards 2 seconds. Find the best frame within those 2 seconds
        for x in range(int(2*vid.get(5))):
            prev_tr_corner = tr_corner
            vid.set(1,vid.get(1)-2)
            _,frame = vid.read()
            tr_corner = frame[0:100, int(vid.get(3)-150):int(vid.get(3))]
            
            # See if the bitwise XOR of this frame and the last frame has the
            # menu buttons in the top-right corner
            diff = cv2.bitwise_xor(tr_corner,prev_tr_corner)
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            diff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,0)
            diffMatch = cv2.matchTemplate(diff, template3, cv2.TM_CCORR_NORMED, template3)
            _,tmpVal,_,_ = cv2.minMaxLoc(diffMatch)
            if (tmpVal > maxVal):
                maxVal = tmpVal
                bestFrame = frame
                bestFrameNumber = vid.get(1)
                #print("bestFrameNumber:"+str(bestFrameNumber))
        filename4 = 'z_'+str(int(bestFrameNumber))+'_enter1.jpg'
        cv2.imwrite(filename4,bestFrame)
        vid.set(1,bestFrameNumber+1)
        _,frame = vid.read()
        filename5 = 'z_'+str(int(bestFrameNumber+1))+'_enter2.jpg'
        cv2.imwrite(filename5,frame)
        print('Player entered hole on frame '+str(bestFrameNumber+1)+
              '. saved images to '+filename4+' and '+filename5)
        
        # Go back to the frame the between-level menu appeared and go forward 1 sec
        vid.set(1,int(round(placeholder+vid.get(5))))
        

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time/60)
seconds = int(elapsed_time%60)
print('Total time: '+str(minutes)+' minutes, '+str(seconds) + ' seconds')
'''
                          ***************************
                          * Useful OpenCV functions *
                          ***************************

***** For VideoCapture object vid *****

vid.get(1) - current frame (float)
vid.get(3) - vid width
vid.get(4) - vid height
vid.get(5) - FPS
vid.get(7) - total frames (float)
vid.set(x,y) - sets vid parameter x to y
    - e.g. vid.set(1,50) sets the current frame to 50
ret, frame = vid.read()
    - ret is True if cv2 stores the picture of the next frame in var frame

***** Basic cv2 functions *****
img = cv2.imread('filename.ext')
    - opens 'filename.ext' and saves the image to var img (must be in same directory)
cv2.imwrite('filename.ext', image)
    - writes image to new file 'filename.ext' in current directory
cv2.imshow('window title', image)
    - shows image in new window.  Useless without cv2.waitKey()
cv2.waitKey()
    - pauses program execution until a key is pressed
cv2.destroyAllWindows()
    - closes all open imshow windows

***** Useful image analysis functions *****
img3 = cv2.sub(img2, img1)
    - For each pixel, the RGB values in img1 is subtracted from the RGB values in img2
      Minimum value is (0,0,0) (color) or 0 (grayscale)
img3 = cv2.bitwise_xor(img1, img2)
    - The only pixels shown are those that are in either img1 or img2 but not both
      Useful to detect the exact frame something appears onscreen

img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    - Converts img1 to grayscale image that can be used with matchTemplate()
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    - Converts img1 to color image that can be used with matchTemplate()
ret,img2 = cv2.threshold(img1, thresh, maxVal, cv2.THRESH_BINARY)
    - Sets all pixels with color values above thresh to maxVal
img2 = cv2.adaptiveThreshold(img1, maxVal, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
    - Sets pixels to maxVal if, within a blocksize x blocksize square, 
      threshold - C is calculated to be high enough
img2 = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED <, mask>)
    - returns a grayscale image showing, at each pixel value, the degree to which
      img resembles template.  100% resemblance is 1.0.  0% resemblance is 0
minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(img)
    - Saves the minimum and maximum values, as well as their respective locations.
      Good for use with matchTemplate()
'''