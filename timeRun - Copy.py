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

filename = 'msbmteam_Shield_Land_Gold.mp4'
print('Opening file '+str(filename))
vid = cv2.VideoCapture(filename)

print('Adjusting templates to match video resolution')
vid.set(1,int(vid.get(7)/2))
templateStar = cv2.imread('star_template.png')
templateStar = cv2.cvtColor(templateStar, cv2.COLOR_BGR2GRAY)

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
bestTemp = templateStar
bestTempMatch = actionFrame
starLoc = (0,0)

actionFrameEdges = cv2.Canny(actionFrame,5,10)
for mult in np.linspace(0.5,1.5,30):
    hTmp = int(round(templateStar.shape[1]*mult))
    wTmp = int(round(templateStar.shape[0]*mult))
    template = cv2.resize(templateStar,(hTmp,wTmp))
    tempMatch = cv2.matchTemplate(actionFrameEdges, template, cv2.TM_CCORR_NORMED, template)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    if maxVal > bestStarMaxVal:
        bestStarMaxVal = maxVal
        bestMult = mult
        bestTemp = template
        starLoc = maxLoc
        bestTempMatch = tempMatch

cv2.imwrite('tmp.png',actionFrameEdges)

templateStar = bestTemp
templateTRCorner = cv2.imread('tr_corner_template.png')
templateTRCorner = cv2.cvtColor(templateTRCorner, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateTRCorner.shape[1]*bestMult))
wTmp = int(round(templateTRCorner.shape[0]*bestMult))
templateTRCorner = cv2.resize(templateTRCorner, (hTmp,wTmp))

templateHoles = cv2.imread('holes_template.png')
templateHoles = cv2.cvtColor(templateHoles, cv2.COLOR_BGR2GRAY)
hTmp = int(round(templateHoles.shape[1]*bestMult))
wTmp = int(round(templateHoles.shape[0]*bestMult))
templateHoles = cv2.resize(templateHoles, (hTmp,wTmp))

tempMatch = cv2.matchTemplate(actionFrameEdges, templateTRCorner, cv2.TM_CCORR_NORMED)
_, TRCornerMaxVal, _, TRCornerMaxLoc = cv2.minMaxLoc(tempMatch)

print('Video is ' + str(bestMult) + ' times as big as a 720p video (confidence:' + str(bestStarMaxVal)+')')
print('Setup complete. Timing the run:')
vid.set(1,-1)
holesLocationFound = False
holesLocation = (0,0)
holeCount = 0
frameCount = 0
vidWriter = cv2.VideoWriter('out.avi',  cv2.cv.FOURCC(*'XVID'), vid.get(5), (int(vid.get(3)),int(vid.get(4))),1)

while holeCount < 9:
    _, frame = vid.read()
    starArea = frame[starLoc[1]:starLoc[1]+templateStar.shape[1]*2,starLoc[0]:starLoc[0]+templateStar.shape[0]*2]
    starArea = cv2.Canny(starArea,100,200)
    tempMatch = cv2.matchTemplate(starArea, templateStar, cv2.TM_CCORR_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
    
    if maxVal > 0.7 * bestStarMaxVal:
        maxVal = 1.0
        holeCount+=1
        print('\nHole '+str(holeCount))
        print('Player became actionable on frame '+str(int(vid.get(1))))
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.waitKey()
            cv2.destroyAllWindows()
        while maxVal >= 0.7*TRCornerMaxVal:
            frameCount+=1
            vidWriter.write(frame)
            _, frame = vid.read()
            tr_corner = frame[TRCornerMaxLoc[1]:TRCornerMaxLoc[1]+templateTRCorner.shape[1]*2,TRCornerMaxLoc[0]:TRCornerMaxLoc[0]+templateTRCorner.shape[0]*2]
            tr_corner = cv2.Canny(tr_corner,5,10)
            tempMatch = cv2.matchTemplate(tr_corner, templateTRCorner, cv2.TM_CCORR_NORMED, templateTRCorner)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(tempMatch)
        print('Player entered hole in frame '+str(int(vid.get(1))))
        frameCount+=1
        vidWriter.write(frame)
        if DEBUG_MODE:
            cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
            cv2.waitKey()
            cv2.destroyAllWindows()
        if holeCount < 9:
            maxVal = 1
            prevMaxVal = 1
            holeLoc = (0,0)
            _,frame = vid.read()
            notBlack = True
            while maxVal/(prevMaxVal+.001) < 1.1 and notBlack:
                prevFrame = frame
                prevMaxVal = maxVal
                _, frame = vid.read()
                frameEdges = cv2.Canny(frame, 100,200)
                tempMatch = cv2.matchTemplate(frameEdges, templateHoles, cv2.TM_CCORR_NORMED)
                minVal, maxVal, minLoc, holeLoc = cv2.minMaxLoc(tempMatch)
                blackArea = frame[int(vid.get(4)/2-50):int(vid.get(4)/2+50),int(vid.get(3)/3):int(vid.get(3)*2/3)]
                blackArea = cv2.cvtColor(blackArea, cv2.COLOR_BGR2GRAY)
                minValB, maxValB, _, _ = cv2.minMaxLoc(blackArea)
                if abs(minValB - maxValB) < 4:
                    notBlack = False
            print('Menu appeared on frame '+str(int(vid.get(1))))
            if DEBUG_MODE:
                cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
                cv2.waitKey()
                cv2.destroyAllWindows()
            minVal = 0
            maxVal = 255
            while abs(minVal-maxVal) >= 4:
                frameCount+=1
                vidWriter.write(frame)
                blackArea = frame[int(vid.get(4)/2+50):int(vid.get(4)/2+100),int(vid.get(3)/2+50):int(vid.get(3)/2+100)]
                _,frame = vid.read()
                blackArea = cv2.cvtColor(blackArea, cv2.COLOR_BGR2GRAY)
                minVal, maxVal,_,_ = cv2.minMaxLoc(blackArea)
            print('The next hole began loading on frame '+str(int(vid.get(1))))
            if DEBUG_MODE:
                cv2.imshow('frame',cv2.resize(frame,(int(vid.get(3)*.5),int(vid.get(4)*.5))))
                cv2.waitKey()
                cv2.destroyAllWindows()
        else:
            print('\nFinished timing the hole!')
            print('Time without loads: '+str(round(frameCount/vid.get(5),3))+' seconds')
            vidWriter.release()
            break
