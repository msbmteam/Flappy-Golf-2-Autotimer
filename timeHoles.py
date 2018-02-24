'''
Flappy Golf Individual Level Auto-Timer
by msbmteam

Procedure:
    - Identify first actionable frame
    - Identify frame of first flap
    - (x8)
        - Identify the frame that the player enters the hole
        - Identify the frame that the between-level menu appears
        - Identify the first actionable frame of the next hole
    - Identify the frame that the player enters the last hole
'''

import cv2
import numpy as np

print('Timing run using video '+filename)
print('Analyzing '+str(int(vid.get(7)))+' frames.  Please wait...')

#Set up templates
tr_corner_template = cv2.imread('tr_corner_template.jpg')
star_template = cv2.imread('star_template.jpg')

#Convert to grayscale (needed for OpenCV's template matching function)
tr_corner_template = cv2.cvtColor(tr_corner_template, cv2.COLOR_BGR2GRAY)
star_template = cv2.cvtColor(star_template, cv2.COLOR_BGR2GRAY)

#Set up important variables
filename = 'FG2 Shield Land Gold.mp4'
vid = cv2.VideoCapture(filename)
if filename[-3:].lower() != 'mp4':
    raise Exception('video format ' + filename[-3:]+' not supported.  Please convert to MP4')

_, last_frame = vid.read()
last_max_val = 0
hole_started = False
hole_count = 0
capture_images = False
time_start_on_flap = False

#The main loop
while vid.get(1) < vid.get(7)-1:
    _, frame = vid.read()
    
    #cv2.imwrite('zref'+str(int(vid.get(1)))+'.jpg', frame)
    
    #Search for (gold) star on the top of the screen
    top = frame[0:100,int(vid.get(3)/3):int(2*vid.get(3)/3)]
    top = cv2.cvtColor(top,cv2.COLOR_BGR2GRAY)
    top = cv2.adaptiveThreshold(top,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
    tempMatch = cv2.matchTemplate(top, star_template, cv2.TM_CCORR_NORMED, star_template)
    _,max_val_star,_,_ = cv2.minMaxLoc(tempMatch)
    
    #If the star is found, the game is in an actionable state. Start timing the hole
    if max_val_star > 0.7 and not(hole_started):
        print('First actionable frame: frame ' + str(int(vid.get(1))))
        if capture_images:
            cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_1.jpg',last_frame)
            cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state0_2.jpg',frame)
        
        hole_started = True
        hole_start_frame = int(vid.get(1))
        
        #Determine which hole this is
        top = frame[0:150,0:int(vid.get(3)/4)]
        top = cv2.cvtColor(top,cv2.COLOR_BGR2GRAY)
        top = cv2.adaptiveThreshold(top,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
        highest_maxVal = 0
        n = 0
        tl_corner = top[0:150, 0:150]
        for x in range (1,10):
            templateN = cv2.cvtColor(cv2.imread(str(x)+'_template.jpg'),cv2.COLOR_BGR2GRAY)
            tempMatch = cv2.matchTemplate(tl_corner, templateN, method=cv2.TM_CCORR_NORMED)
            minVal, max_val_star, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
            if max_val_star > highest_maxVal and max_val_star > 0.5:
                highest_maxVal = max_val_star
                n = x
        
        #Identify the player's first frame of movement (only for Hole 1)
        if time_start_on_flap and n == 1:
            frametmp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[85:int(vid.get(4)),0:int(vid.get(3))]
            frametmp = cv2.adaptiveThreshold(frametmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-10)
            tempMatch = cv2.matchTemplate(frametmp, template3, cv2.TM_CCORR_NORMED, template3)
            tempMatchIMG = cv2.normalize(tempMatch, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            _,maxVal,_,location0 = cv2.minMaxLoc(tempMatch)
            location0[1]
            cv2.rectangle(frame, (location0[0], location0[1]+85), (location0[0]+25, location0[1]+85+25), (0,0,255),1)
            print(location0)
            print(location0[0])
            print(location0[1])
            
            while maxVal > 0.7:
                last_frame = frame
                _, frame = vid.read()
                frame = frame[location0[1]-25+85:location0[1]+50+85,location0[0]-25:location0[0]+50]
                frametmp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frametmp = cv2.adaptiveThreshold(frametmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)
                tempMatch = cv2.matchTemplate(frametmp, template3, cv2.TM_CCORR_NORMED, template3)
                tempMatchIMG = cv2.normalize(tempMatch, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
            print('First frame of movement: frame ' + str(int(vid.get(1))))
        
        #Identify the frame that the player goes in the hole and the frame that the inter-level menu appears
        
        retry_visible = True
        while retry_visible:
            _, frame = vid.read()
    
            tr_corner = frame[0:100,int(vid.get(3))-200:int(vid.get(3))]
            tr_corner = cv2.cvtColor(tr_corner,cv2.COLOR_BGR2GRAY)
            tr_corner = cv2.adaptiveThreshold(tr_corner,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,0)
            tempMatch = cv2.matchTemplate(tr_corner, tr_corner_template, method=cv2.TM_CCORR_NORMED)
            minVal, max_val_star, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
        
            if max_val_star-last_max_val < -0.05 and max_val_star > 0.1 and max_val_star/(.01+last_max_val) < 0.75:
                print('    retry is not visible in frame ' + str(int(vid.get(1))) + ' (confidence:' + str(round(max_val_star-last_max_val,2))+', maxVal:'+str(round(max_val_star,2))+', quotient:'+str(round(max_val_star/(.01+last_max_val),2))+')')
                cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state1_1.jpg',last_frame)
                cv2.imwrite('zframe_'+str(int(vid.get(1)))+'state1_2.jpg',frame)
                retry_visible = False
            else:
                #print('    retry frame rejected ' + str(int(vid.get(1))) + ' (confidence:' + str(round(max_val_star-last_max_val,2))+', maxVal:'+str(round(max_val_star,2))+', quotient:'+str(round(max_val_star/(.01+last_max_val),2))+')')
                retry_visible = True
                
            last_max_val = max_val_star
            last_frame = frame
        hole_end_frame = int(vid.get(1))
        hole_time = hole_end_frame-hole_start_frame+1
        
        tl_corner = frame[0:100,0:int(3*vid.get(3)/4)]
        tl_corner = cv2.cvtColor(tl_corner,cv2.COLOR_BGR2GRAY)
        tl_corner = cv2.adaptiveThreshold(tl_corner,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,49,0)
        tempMatch = cv2.matchTemplate(tl_corner, star_template, method=cv2.TM_CCORR_NORMED)
        minVal, max_val_star, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
        if (not(retry_visible) and max_val_star > 0.7):
            print('Hole ' + str(n) + ' in ' + str(hole_time) + ' frames = ' + str(round(hole_time/vid.get(5),3)) + ' seconds')
    else:
        hole_started = False
        
    
    last_frame = frame