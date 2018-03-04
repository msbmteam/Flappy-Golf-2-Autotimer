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

DEBUG_MODE = False
FRAME_BY_FRAME = False

filename = 'Ayrton_Golf_Land.mp4'
print('Opening file '+str(filename))
vid = cv2.VideoCapture(filename)

print('Adjusting templates to match video resolution')
vid.set(1,int(vid.get(7)/2))
templateStarCanny = cv2.imread('star_template_canny.png')
templateStarCanny = cv2.cvtColor(templateStarCanny, cv2.COLOR_BGR2GRAY)

_, frame = vid.read()
center = frame[int(vid.get(4)/2-50):int(vid.get(4)/2),int(vid.get(3)/2+50):int(vid.get(3)/2+100)]
center = cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(center)

while abs(maxVal-minVal) > 4:
    _, frame = vid.read()
    center = frame[int(vid.get(4)/2-50):int(vid.get(4)/2),int(vid.get(3)/2+50):int(vid.get(3)/2+100)]
    center = cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(center)
frame1 = vid.get(1)

while abs(maxVal-minVal) <= 4:
    _, frame = vid.read()
    center = frame[int(vid.get(4)/2-50):int(vid.get(4)/2),int(vid.get(3)/2+50):int(vid.get(3)/2+100)]
    center = cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(center)
    
while abs(maxVal-minVal) > 4:
    _, frame = vid.read()
    center = frame[int(vid.get(4)/2-50):int(vid.get(4)/2),int(vid.get(3)/2+50):int(vid.get(3)/2+100)]
    center = cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(center)
frame2 = vid.get(1)

vid.set(1,(frame1+frame2)/2)
_, actionFrame = vid.read()

bestMult = 1
bestStarMaxVal = 0
bestTemp = templateStarCanny
bestTempMatch = actionFrame
starLoc = (0,0)

actionFrameEdges = cv2.Canny(actionFrame,5,10)
for mult in np.linspace(0.5,1.5,30):
    hTmp = int(round(templateStarCanny.shape[1]*mult))
    wTmp = int(round(templateStarCanny.shape[0]*mult))
    template = cv2.resize(templateStarCanny,(hTmp,wTmp))
    tmp = actionFrameEdges[0:int(vid.get(4)/4), 0:int(vid.get(3)*2/3)]
    tempMatch = cv2.matchTemplate(tmp, template, cv2.TM_CCORR_NORMED, template)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    if maxVal > bestStarMaxVal:
        bestStarMaxVal = maxVal
        bestMult = mult
        bestTemp = template
        starLoc = maxLoc
        bestTempMatch = tempMatch
if DEBUG_MODE:
    cv2.imshow('actionFrameEdges',actionFrame)
    cv2.imshow('resized template',bestTemp)
    cv2.imshow('original template',templateStarCanny)
    cv2.imshow('matchTemp',bestTempMatch)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
templateStarCanny = bestTemp
    
templateStar = cv2.imread('star_template.png')
templateStar = cv2.cvtColor(templateStar, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateStar.shape[1]*bestMult))
wTmp = int(round(templateStar.shape[0]*bestMult))
templateStar = cv2.resize(templateStar, (hTmp,wTmp))

templateTRCorner = cv2.imread('tr_corner_template2.png')
templateTRCorner = cv2.cvtColor(templateTRCorner, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateTRCorner.shape[1]*bestMult))
wTmp = int(round(templateTRCorner.shape[0]*bestMult))
templateTRCorner = cv2.resize(templateTRCorner, (hTmp,wTmp))

templateHoles = cv2.imread('holes_template.png')
templateHoles = cv2.cvtColor(templateHoles, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateHoles.shape[1]*bestMult))
wTmp = int(round(templateHoles.shape[0]*bestMult))
templateHoles = cv2.resize(templateHoles, (hTmp,wTmp))

templateTRCornerCanny = cv2.imread('tr_corner_template_canny.png')
templateTRCornerCanny = cv2.cvtColor(templateTRCornerCanny, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateTRCornerCanny.shape[1]*bestMult))
wTmp = int(round(templateTRCornerCanny.shape[0]*bestMult))
templateTRCornerCanny = cv2.resize(templateTRCornerCanny, (hTmp,wTmp))

# actionFrameEdges2 = cv2.cvtColor(actionFrame,cv2.COLOR_BGR2GRAY)
# actionFrameEdges2 = cv2.adaptiveThreshold(actionFrameEdges2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,10)

tempMatch = cv2.matchTemplate(actionFrameEdges, templateTRCornerCanny, cv2.TM_CCORR_NORMED)
_, TRCornerMaxVal, _, TRCornerMaxLoc = cv2.minMaxLoc(tempMatch)
if DEBUG_MODE:
    cv2.imshow('tempmatch',tempMatch)
    cv2.imshow('actionFrameEdges',actionFrameEdges)
    cv2.imwrite('tmp.png',actionFrameEdges)
    cv2.waitKey()
    cv2.destroyAllWindows()

print('Video is about' + str(round(bestMult,2)) + ' times as big as a 720p video')
print('Setup complete. Timing the run:')
vid.set(1,-1)
holesLocationFound = False
holesLoc = (0,0)
holeCount = 0
bookendFrames = []

_, frame = vid.read()
starArea = frame[starLoc[1]:int(starLoc[1]+templateStar.shape[0]),
                 starLoc[0]-templateStar.shape[0]:starLoc[0]+templateStar.shape[1]*2]
starArea = cv2.Canny(starArea,100,200)
maxVal = 1
while holeCount < 9:
    lastStarArea = starArea
    lastMaxVal = maxVal
    _, frame = vid.read()
    starArea = frame[starLoc[1]:int(starLoc[1]+templateStar.shape[0]),
                     starLoc[0]-templateStar.shape[0]:starLoc[0]+templateStar.shape[1]*2]
    starArea = cv2.Canny(starArea,100,200)
    starXOR = cv2.bitwise_xor(starArea,lastStarArea)
    tempMatch = cv2.matchTemplate(starXOR, templateStarCanny, cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    if DEBUG_MODE and FRAME_BY_FRAME:
        print(maxVal, bestStarMaxVal)
        cv2.imshow('frame',frame)
        cv2.imshow('tempMatch',tempMatch)
        cv2.imshow('starArea',starArea)
        cv2.waitKey()
    
    if maxVal/(lastMaxVal+.001) > 1.2 and maxVal > 0.8*bestStarMaxVal:
        holeCount+=1
        print('\nHole '+str(holeCount))
        print('Player became actionable on frame '+str(int(vid.get(1))))
        bookendFrames.append(int(vid.get(1)))
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.imshow('starXOR',starXOR)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        vid.set(1,vid.get(1)+int(round(vid.get(5)))*2)
        _, frame = vid.read()
        holesArea = frame[0:int(vid.get(4)/2),0:int(vid.get(3)/2)]
        if holesLocationFound:
            holesArea = frame[holesLoc[1]:int(holesLoc[1]+templateHoles.shape[0]),
                              holesLoc[0]:int(holesLoc[0]+templateHoles.shape[1])]
        holesArea = cv2.cvtColor(holesArea,cv2.COLOR_BGR2GRAY)
        holesArea = cv2.adaptiveThreshold(holesArea,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
        tempMatch = cv2.matchTemplate(holesArea, templateHoles, method=cv2.TM_CCORR_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
        lastMaxVal = maxVal
        while maxVal < 0.875:
            # See if the next frame has the between-level menu visible (specifically looks for holes button)
            lastFrame = frame
            lastMaxVal = maxVal
            _, frame = vid.read()
            holesArea = frame[0:int(vid.get(4)/2),0:int(vid.get(3)/2)]
            if holesLocationFound:
                holesArea = frame[holesLoc[1]:int(holesLoc[1]+templateHoles.shape[0]),
                                  holesLoc[0]:int(holesLoc[0]+templateHoles.shape[1])]
            holesArea = cv2.cvtColor(holesArea,cv2.COLOR_BGR2GRAY)
            holesArea = cv2.adaptiveThreshold(holesArea,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
            tempMatch = cv2.matchTemplate(holesArea, templateHoles, method=cv2.TM_CCORR_NORMED)
            minVal, maxVal, minLoc, holesLocTmp = cv2.minMaxLoc(tempMatch)
            if not(holesLocationFound):
                holesLoc = holesLocTmp
            
            if DEBUG_MODE and FRAME_BY_FRAME and holeCount == 7:
                cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
                cv2.imshow('tempMatch',tempMatch)
                cv2.waitKey()
                cv2.destroyAllWindows()
                print(lastMaxVal, maxVal)
        holesLocationFound = True
        
            
        print('menu appeared at frame ' + str(int(vid.get(1))))
        bookendFrames.append(int(vid.get(1)))
        
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.imshow('tempMatch',tempMatch)
            cv2.waitKey()
            cv2.destroyAllWindows()
        placeholder = vid.get(1)
        
        # Go backward frame-by-frame to find the frame the pause/retry buttons in 
        # the top-right corner appear onscreen
        vid.set(1,vid.get(1)-2)
        _,frame = vid.read()
        tr_corner = frame[TRCornerMaxLoc[1]:TRCornerMaxLoc[1]+templateTRCorner.shape[0],
                          TRCornerMaxLoc[0]:TRCornerMaxLoc[0]+templateTRCorner.shape[1]]
        maxVal = 0
        diff = tr_corner
        # Go backwards 2 seconds. Find the best frame within those 2 seconds
        for x in range(int(2*vid.get(5))):
            prev_tr_corner = tr_corner
            vid.set(1,vid.get(1)-2)
            _,frame = vid.read()
            tr_corner = frame[TRCornerMaxLoc[1]:TRCornerMaxLoc[1]+templateTRCorner.shape[0],
                              TRCornerMaxLoc[0]:TRCornerMaxLoc[0]+templateTRCorner.shape[1]]
            
            # See if the bitwise XOR of this frame and the last frame has the
            # menu buttons in the top-right corner
            diff = cv2.bitwise_xor(tr_corner,prev_tr_corner)
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            diff = cv2.adaptiveThreshold(diff,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,0)
            diffMatch = cv2.matchTemplate(diff, templateTRCorner, cv2.TM_CCORR_NORMED,templateTRCorner)
            _,tmpVal,_,_ = cv2.minMaxLoc(diffMatch)
            if (tmpVal > maxVal):
                maxVal = tmpVal
                bestFrame = frame
                bestFrameNumber = vid.get(1)
                #print("bestFrameNumber:"+str(bestFrameNumber))
        print('Player entered hole on frame '+str(bestFrameNumber+1))
        bookendFrames = bookendFrames[:-1] + [int(bestFrameNumber+1)] + [bookendFrames[-1]]        
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(bestFrame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.imshow('prevCorner',prev_tr_corner)
            cv2.imshow('corner',tr_corner)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if holeCount >= 9:
            break
        
        vid.set(1,int(round(placeholder-1)))
        _,frame = vid.read()
        minVal = 0
        maxVal = 255

        while abs(minVal-maxVal) >= 4:
            _,frame = vid.read()
            holesArea = frame[holesLoc[1]:int(holesLoc[1]+templateHoles.shape[0]),
                              holesLoc[0]:int(holesLoc[0]+templateHoles.shape[1])]
            holesArea = cv2.cvtColor(holesArea, cv2.COLOR_BGR2GRAY)
            minVal, maxVal,_,_ = cv2.minMaxLoc(holesArea)
            
        print('The next hole began loading on frame '+str(int(vid.get(1))))
        bookendFrames.append(int(vid.get(1))-1)
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.waitKey()
            cv2.destroyAllWindows()
        while abs(minVal-maxVal) < 4:
            _,frame = vid.read()
            holesArea = frame[holesLoc[1]:int(holesLoc[1]+templateHoles.shape[0]),
                              holesLoc[0]:int(holesLoc[0]+templateHoles.shape[1])]
            holesArea = cv2.cvtColor(holesArea, cv2.COLOR_BGR2GRAY)
            minVal, maxVal,_,_ = cv2.minMaxLoc(holesArea)
        starArea = frame[starLoc[1]:int(starLoc[1]+templateStar.shape[0]),
                         starLoc[0]-templateStar.shape[0]:starLoc[0]+templateStar.shape[1]*2]
        starArea = cv2.Canny(starArea,100,200)
        print(bookendFrames)
print('\nFinished identifying load times!')
outFileName = filename[:-4]+' [no loads].avi'
fourcc = cv2.cv.FOURCC(*'XVID')
print('Creating video...')

vidWriter = cv2.VideoWriter(outFileName, fourcc, vid.get(5), (int(vid.get(3)),int(vid.get(4))),1)
vid.set(1,-1)
bookendIndex = 0
while bookendIndex < len(bookendFrames):
    vid.set(1,bookendFrames[bookendIndex]-1)
    print(str(round(vid.get(1)/bookendFrames[-1]*100,0)) + ' percent done')
    bookendIndex += 1
    while bookendIndex < len(bookendFrames) and vid.get(1) < bookendFrames[bookendIndex]:
        _,frame = vid.read()
        vidWriter.write(frame)
    bookendIndex += 1
vidWriter.release()

print ('Done! Loadless run saved to '+outFileName)
