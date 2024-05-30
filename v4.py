import os
# comment out below line to enable tensorflow logging outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imutils
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


from _collections import deque
import gc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score
import statistics as st


def intialize():
    global missed,m,cen0,cen1,next_cen0,next_cen1,ratio,iou_ratio,dir,dir1,depth,timer,height,class_name,skip,ptsGt,ptsExp30,ptsExp60,ptsExp90,toler,area_Rate,tolerx,tolery
    global prev,pres_Area,prev_Area,prev_Avg_Area,pres_X,prev_X,prev_Avg_X,steps,speed,pts,counter,t1,ptsExp120,ptsExp150,ptsExp180,tolerance,mul,vid,out,k,ptsG,ptsE
    global area_Rate, prev_area_Rate,ids
    m=4
    t1=0
    timer=0
    height=0
    mul=2
    skip=5
    class_name=''
    ptsG=[]*0
    ptsE=[]*0
    ids=[]*0
    cen0 = [deque(maxlen=m) for i in range(1000)] #center X for the pedestrian before 5 frames
    cen1 = [deque(maxlen=m) for i in range(1000)] #center Y for the pedestrian before 5 frames
    next_cen0 = [0 for i in range(1000)] #Sum of pairwise adjacent X center difference before 5 frames for predicting next bbox
    next_cen1 = [0 for i in range(1000)] #Sum of pairwise adjacent Y center difference before 5 frames for predicting next bbox
    ratio = [0 for i in range(1000)] #Occulsion IOU ratio with other neighbouring bounding boxes
    iou_ratio = [0 for i in range(1000)] #Sum of Occulsion IOU ratio with other neighbouring bounding boxes
    missed = [0 for i in range(1000)] #If the pedestrain goes missing for a frame and comes back in next frame
    dir = [deque(maxlen=30) for i in range(1000)] #Direction of the pedestarin
    dir1 = [0 for i in range(1000)] #Direction of the pedestarin
    depth = [0 for i in range(1000)] #distance from camera
    toler=[]*0
    tolerx=[]*0
    tolery=[]*0
    ptsExp180 = [[]*0 for i in range(1000)]
    ptsExp150 = [[]*0 for i in range(1000)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp120 = [[]*0 for i in range(1000)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp90 = [[]*0 for i in range(1000)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp60 = [[]*0 for i in range(1000)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp30 = [[]*0 for i in range(1000)]
    ptsGt = [[]*0 for i in range(1000)]
    #prev_depth = [0 for i in range(10)] #previous cumulative distance from camera
    #prev_Avg_depth = [0 for i in range(10)] #Averege distance from camera
    #pres_Y = [0 for i in range(10)] #Center Y of present frame bounding box 
    #prev_Y = [0 for i in range(10)] #Center Y of previous frame bounding box 
    #prev_Avg_Y = [1 for i in range(10)] #Mean of Center Y of previous frames bounding box
    #prev_h = [0 for i in range(10)] #Center Y of previous frame bounding box 
    #prev_Avg_h = [1 for i in range(10)] #Mean of Center Y of previous frames bounding box
    #dirpt = [] #Direction of the pedestarin predicted
    prev = [deque(maxlen=18) for _ in range(1000)] #Previous Areas of each bounding boxes from atmost past 30 frames
    area_Rate = [1 for _ in range(1000)] #Previous Areas of each bounding boxes from atmost past 30 frames
    prev_area_Rate = [deque(maxlen=18) for _ in range(1000)] #Previous Areas of each bounding boxes from atmost past 30 frames
    pres_Area = [0 for i in range(1000)] #Area of present frame bounding box 
    prev_Area = [0 for i in range(1000)] #Area of previous frame bounding box 
    prev_Avg_Area = [1 for i in range(1000)] #Mean of areas of previous frames bounding box
    pres_X = [0 for i in range(1000)] #Center X of present frame bounding box 
    prev_X = [0 for i in range(1000)] #Center X of previous frame bounding box 
    prev_Avg_X = [1 for i in range(1000)] #Mean of Center X of previous frames bounding box
    steps = [0 for i in range(1000)] #Steps walked so far
    speed = [0 for i in range(1000)] #Speed of the pedestrain
    pts = [deque(maxlen=6) for _ in range(1000)] #Centroids of each bounding boxes from atmost past 30 frames
    counter = [] #Counts number of pedestrains crossing the barrier

def iou(bbox, candidates):
    """Helps in Computing parametrs for modified intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray's
    area_intersection, area_candidates, area_bbox

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection, area_candidates, area_bbox

def Occlusion_rate(track,tracker,area_Rate,acc):
            global cand_id,Occulsion
            
            #11. Predecting the next bounding box for the pedestrian after 5 frames
            cen0[track.track_id].append(center[0]) #Appending center X points
            cen1[track.track_id].append(center[1]) #Appending center Y points

            '''
            #Sum of pairwise adjacent difference for predicting
            dif = np.diff(cen0[track.track_id])
            s = 0
            for d in dif:
                s = d + area_Rate * s
            next_cen0[track.track_id]=int(mul*s)
            dif = np.diff(cen1[track.track_id])
            s = 0
            for d in dif:
                s = d + area_Rate * s
            next_cen1[track.track_id]=int(mul*s)
            '''

            if(len(abs(np.diff(cen0[track.track_id])))!=0):
                next_cen0[track.track_id]=int(mul*sum(np.diff(cen0[track.track_id]))) 
                next_cen1[track.track_id]=int(mul*sum(np.diff(cen1[track.track_id])))

            if count>=30:
                ptsGt[track.track_id].append((center[0],center[1]))
                ptsExp30[track.track_id].append((center[0]+(area_Rate+acc)*next_cen0[track.track_id],(center[1]+(area_Rate+acc)*next_cen1[track.track_id])))
                ptsExp60[track.track_id].append((center[0]+(area_Rate+(acc*2))*(2*next_cen0[track.track_id]),(center[1]+(area_Rate+(acc*2))*(2*next_cen1[track.track_id]))))
                ptsExp90[track.track_id].append((center[0]+(area_Rate+(acc*3))*(3*next_cen0[track.track_id]),(center[1]+(area_Rate+(acc*3))*(3*next_cen1[track.track_id]))))
                ptsExp120[track.track_id].append((center[0]+(area_Rate+(acc*4))*(4*next_cen0[track.track_id]),(center[1]+(area_Rate+(acc*4))*(4*next_cen1[track.track_id]))))
                ptsExp150[track.track_id].append((center[0]+(area_Rate+(acc*5))*(5*next_cen0[track.track_id]),(center[1]+(area_Rate+(acc*5))*(5*next_cen1[track.track_id]))))
                ptsExp180[track.track_id].append((center[0]+(area_Rate+(acc*6))*(6*next_cen0[track.track_id]),(center[1]+(area_Rate+(acc*6))*(6*next_cen1[track.track_id]))))
            

            id=track.track_id
            #Selecting the neighbour candidates(tracks) bounding boxes of our PRESENT track if track is confirmed and has new changes and not present track and class aeroplane
            candidates = np.asarray([(track.to_tlwh()[0]+(area_Rate+(acc*6))*(6*next_cen0[track.track_id]), track.to_tlwh()[1]+(area_Rate+(acc*6))*(6*next_cen1[track.track_id]), track.to_tlwh()[2], track.to_tlwh()[3]) for track in tracker.tracks if track.is_confirmed() and track.time_since_update<=1 and track.track_id!=id and track.get_class() == 'aeroplane'])
            bbox = np.asarray((track.to_tlwh()[0]+(area_Rate+(acc*6))*(6*next_cen0[track.track_id]), track.to_tlwh()[1]+(area_Rate+(acc*6))*(6*next_cen1[track.track_id]), track.to_tlwh()[2], track.to_tlwh()[3])) #(top left x, top left y, width, height)

            if(len(candidates)!=0):

                area_intersection, area_candidates, area_bbox=iou(bbox,candidates)
                cand_id = np.asarray([track.track_id for track in tracker.tracks if track.is_confirmed() and track.time_since_update<=1 and track.track_id!=id and track.get_class() == 'aeroplane'])
                cand_id=cand_id[np.nonzero(area_intersection)]
                #Selecting the neighbour candidates(tracks) area of our PRESENT track if PRESENT track is actually intersected to the neighbour candidates
                area_candidates = np.asarray([area for area in area_candidates if area_intersection[np.where(area_candidates==area)]!=0])
                area_intersection=area_intersection[np.nonzero(area_intersection)]

                #3. Calculating Modified IOU
                '''
                The ratio in [0, 1] between the `bbox` and each
                candidate. A higher score means a larger fraction of the `bbox` is
                occluded by the candidate.
                '''
                
                if(len(area_candidates)!=0):
                    #3.a For Relatively Smaller Bounding Boxes 
                    if(area_bbox<np.mean(area_candidates)):
                        #If our PRESENT track area is less than average of its neighbour candidates(tracks) areas
                        #We wont consider the its bigger neighbour candidates(tracks) area for impartiality
                        ratio[track.track_id]=area_intersection / (2*area_bbox - area_intersection)

                    #3.b For Relatively Bigger Bounding Boxes 
                    else:
                        #If our PRESENT track area is more than average of its neighbour candidates(tracks) areas
                        #We wont consider our bigger PRESENT track area for impartiality
                        ratio[track.track_id]=area_intersection / (2*area_candidates - area_intersection)
                    cand_id=cand_id[np.argmax(ratio[track.track_id])]
                    iou_ratio[track.track_id]=sum(ratio[track.track_id])

                else:

                    iou_ratio[track.track_id]=0
                
                #4. Calculating Occulsion Rate
                if(iou_ratio[track.track_id]>0.3):
                    #If our PRESENT track Occulsion is more than 30%, indicated by RED
                    color=(0,0,255)
                elif(iou_ratio[track.track_id]>0.2):
                    #If our PRESENT track Occulsion is more than 20%, indicated by ORANGE
                    color=(0,165,255)
                else:
                    #If our PRESENT track Occulsion is less than 20%, indicated by GREEN
                    color=(0,255,0)
                    Occulsion=0
                
                #5. Detecting Occludee(1)/Occluder(2)
                if(len(area_candidates)!=0 and iou_ratio[track.track_id]>0.2):
                    
                    '''
                    for track in tracker.tracks:
                        if track.track_id==cand_id:
                            h=track.to_tlwh()[3]
                    if len(cen1[track.track_id]) >0 and len(cen1[cand_id]) >0:
                        if ((cen1[track.track_id][-1]+track.to_tlwh()[3]/2)<(cen1[cand_id][-1]+h/2)):
                            #If our PRESENT track area is less than minimum of its neighbour candidates(tracks) areas
                            #Our PRESENT track is getting Occluded by atleast one of its neighbour candidates(tracks), so its an Occludee
                            Occulsion=1
                        else:
                            #If our PRESENT track area is more than minimum of its neighbour candidates(tracks) areas
                            #Our PRESENT track is Occluding atleast one of its neighbour candidates(tracks), so its an Occluder
                            Occulsion=2
                            color=(255,0,0) #Indicated by BLUE
                    '''
                    
                    if(depth[track.track_id]<depth[cand_id]):
                        #If our PRESENT track area is less than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is getting Occluded by atleast one of its neighbour candidates(tracks), so its an Occludee
                        Occulsion=1
                    else:
                        #If our PRESENT track area is more than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is Occluding atleast one of its neighbour candidates(tracks), so its an Occluder
                        Occulsion=2
                        color=(255,0,0) #Indicated by BLUE

                    '''
                    if(area_bbox<(min(area_candidates))):
                        #If our PRESENT track area is less than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is getting Occluded by atleast one of its neighbour candidates(tracks), so its an Occludee
                        Occulsion=1
                    else:
                        #If our PRESENT track area is more than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is Occluding atleast one of its neighbour candidates(tracks), so its an Occluder
                        Occulsion=2
                        color=(255,0,0) #Indicated by BLUE
                    '''

                #1. Missing aeroplane Detection And Reappearance In The Next Frame
                if(missed[track.track_id]==1 and iou_ratio[track.track_id]>0.2 and Occulsion==1):
                    #Our PRESENT track is getting Occluded by atleast one of its neighbour candidates(tracks), so its an Occludee
                    color=(0,0,0) #Indicted by BLACK 
                    missed[track.track_id]=0 #Not Missing Anymore
            else:
                color=(0,255,0) #If no neighbour candidates(tracks), no occlusion, indiacted by GREEN
                Occulsion=0
            return color

def direction(track):
            global acc
            acc=0
            bbox = track.to_tlbr() #(min x, miny, max x, max y)
            #6. Detecting Approaching(A)/Departing(D)/Left(L)/Right(R)
            pres_X[track.track_id] = int(((bbox[0])+(bbox[2]))/2)
            #pres_Y[track.track_id] = int(((bbox[1])+(bbox[3]))/2)
            pres_Area[track.track_id]=(int(bbox[2])-int(bbox[0]))*(int(bbox[3])-int(bbox[1])) #Width*Height
            if(len(prev[track.track_id])>1 and len(prev_area_Rate[track.track_id])>1):
                area_Rate[track.track_id]=pres_Area[track.track_id]/np.mean(prev[track.track_id])
                #print(area_Rate[track.track_id])
                acc=area_Rate[track.track_id]-prev_area_Rate[track.track_id][0]
                acc=acc/6
                #print(acc)
                #area_Rate=pres_Area[track.track_id]/prev[track.track_id][0]
            if(pres_Area[track.track_id]/prev_Avg_Area[track.track_id]>=1.075):
                #If the area of PRESENT bounding box is increasing relatively, Approaching
                if(pres_X[track.track_id]/prev_Avg_X[track.track_id]<=0.95):
                    #X Coordinate decreasing, going left
                    dir[track.track_id].append("6")
                elif(pres_X[track.track_id]/prev_Avg_X[track.track_id]>1.05):
                    #X Coordinate increasing, going right
                    dir[track.track_id].append("4")
                else:
                    dir[track.track_id].append("5")

            elif(pres_Area[track.track_id]/prev_Avg_Area[track.track_id]<0.925):
                #If the area of PRESENT bounding box is decreasing relatively, Departing
                if(pres_X[track.track_id]/prev_Avg_X[track.track_id]<=0.95):
                    #X Coordinate decreasing, going left
                    dir[track.track_id].append("8")
                elif(pres_X[track.track_id]/prev_Avg_X[track.track_id]>1.05):
                    #X Coordinate increasing, going right
                    dir[track.track_id].append("2")
                else:
                    dir[track.track_id].append("1")

            else:
                #If the area of PRESENT bounding box is almost constant
                if(pres_X[track.track_id]/prev_Avg_X[track.track_id]<=1):
                    #X Coordinate decreasing, going left
                    dir[track.track_id].append("7")
                if(pres_X[track.track_id]/prev_Avg_X[track.track_id]>1):
                    #X Coordinate increasing, going right
                    dir[track.track_id].append("3")
            
            dir1[track.track_id]=st.mode(dir[track.track_id])
            
            
            #Cummulative previous area of bounding box from its birth 
            #prev_Y[track.track_id]=prev_Y[track.track_id]+(int(((bbox[1])+(bbox[3]))/2))
            #prev_Avg_Y[track.track_id]=prev_Y[track.track_id]/track.age #Average cummulative previous Y
            #prev_h[track.track_id]=prev_h[track.track_id]+(track.to_tlwh()[3])
            #prev_Avg_h[track.track_id]=prev_h[track.track_id]/track.age #Average cummulative previous Y
            #prev_depth[track.track_id]=prev_depth[track.track_id]+depth[track.track_id]
            #prev_Avg_depth[track.track_id]=prev_depth[track.track_id]/track.age
            prev[track.track_id].append((track.to_tlwh()[2])*(track.to_tlwh()[3]))
            prev_area_Rate[track.track_id].append(area_Rate[track.track_id])
            prev_Area[track.track_id]=prev_Area[track.track_id]+((track.to_tlwh()[2])*(track.to_tlwh()[3]))
            prev_Avg_Area[track.track_id]=prev_Area[track.track_id]/track.age #Average cummulative previous area
            prev_X[track.track_id]=prev_X[track.track_id]+(int(((bbox[0])+(bbox[2]))/2))
            prev_Avg_X[track.track_id]=prev_X[track.track_id]/track.age #Average cummulative previous X
            return area_Rate[track.track_id],acc

def velocity(track, area_Rate):
    #12. Apparent Speed of pedestrain per second(30 frames)
            speed[track.track_id]=0
            Euclidean=1
            #print(area_Rate)
            if area_Rate<1:
                area_Rate=1/area_Rate
            area_Rate=100*area_Rate
            for j in range(1, len(pts[track.track_id])):
                    #Euclidean distance of bounding box centers between adjacent frames
                    Euclidean=Euclidean+(np.linalg.norm(np.array(pts[track.track_id][j])-np.array(pts[track.track_id][j-1])))
            print(Euclidean)
            alpha=area_Rate/Euclidean
            #speed[track.track_id]=(alpha*area_Rate+(1-alpha)*Euclidean)
            speed[track.track_id]=Euclidean
            '''
            if(dir1[track.track_id]=='5'):
                #Ratio of present bounding box area and mean of the previous 30 bounding box area
                speed[track.track_id]=speed[track.track_id]+100*(pres_Area[track.track_id]/np.mean(prev[track.track_id]))
            elif(dir1[track.track_id]=='1'):
                #Ratio of  mean of the previous 30 bounding box area and present bounding box area 
                speed[track.track_id]=speed[track.track_id]+100*(np.mean(prev[track.track_id])/pres_Area[track.track_id])
            elif(dir1[track.track_id]=='7' or dir1[track.track_id]=='3'):
                for j in range(1, len(pts[track.track_id])):
                    #Euclidean distance of bounding box centers between adjacent frames
                    speed[track.track_id]=speed[track.track_id]+(np.linalg.norm(np.array(pts[track.track_id][j])-np.array(pts[track.track_id][j-1]))/2)
            elif(dir1[track.track_id]=='2' or dir1[track.track_id]=='8'):
                for j in range(1, len(pts[track.track_id])):
                    #Euclidean distance of bounding box centers between adjacent frames*To remove Perspective distortion
                    speed[track.track_id]=speed[track.track_id]+(0.7*(0.25*np.linalg.norm(np.array(pts[track.track_id][j])-np.array(pts[track.track_id][j-1]))+0.75*3*(2-(pres_Area[track.track_id]/(prev_Avg_Area[track.track_id])))))
            else:
                for j in range(1, len(pts[track.track_id])):
                    #Euclidean distance of bounding box centers between adjacent frames*To remove Perspective distortion
                    speed[track.track_id]=speed[track.track_id]+(0.25*np.linalg.norm(np.array(pts[track.track_id][j])-np.array(pts[track.track_id][j-1]))+0.25*7*(prev_Avg_Area[track.track_id]/(pres_Area[track.track_id])))
            

            #13. Steps walked till now
            if(speed[track.track_id]!=0):
                steps[track.track_id]=steps[track.track_id]+speed[track.track_id]/(30*50) #speed/(fps of video*frames needed for single step)
            '''

def display(track,area_Rate,acc,timer):

            h=track.to_tlwh()[3]
            #Adding the bounding box in the image
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            #Adding the pedestrain prodata/video/mixkit-people-in-the-subway-hall-in-tokyo-4454.mp4perties on top of bounding box in the image
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-90)), (int(bbox[0])+(len(str(track.track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-60)), (int(bbox[0])+(len(str(track.track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track.track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, str(track.track_id)+"-"+str(dir1[track.track_id])+"-"+str(round(depth[track.track_id],2)), (int(bbox[0]), int(bbox[1]-70)), 0, 1, (255, 255, 255), 2)
            cv2.putText(img, str(round(speed[track.track_id],1))+"-"+str(round(steps[track.track_id],1)), (int(bbox[0]), int(bbox[1]-40)), 0, 1, (255, 255, 255), 2)
            cv2.putText(img, str(round(timer,1)), (int(bbox[0]), int(bbox[1]-10)), 0, 1, (255, 255, 255), 2)
            
            '''
            cv2.rectangle(img, ((int(bbox[0]+(area_Rate+(acc*1))*1*next_cen0[track.track_id])), (int(bbox[1]+(area_Rate+(acc*1))*1*next_cen1[track.track_id]))), ((int(bbox[2]+(area_Rate+(acc*1))*1*next_cen0[track.track_id])),(int(bbox[3]+(area_Rate+(acc*1))*1*next_cen1[track.track_id]))), (255,255,255), 2)
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*2))*2*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*2))*2*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*2))*2*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*2))*2*next_cen1[track.track_id])), (255,255,255), 2)
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*3))*3*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*3))*3*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*3))*3*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*3))*3*next_cen1[track.track_id])), (255,255,255), 2)
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*4))*4*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*4))*4*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*4))*4*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*4))*4*next_cen1[track.track_id])), (255,255,255), 2)
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*5))*5*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*5))*5*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*5))*5*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*5))*5*next_cen1[track.track_id])), (255,255,255), 2)
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*6))*6*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*6))*6*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*6))*6*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*6))*6*next_cen1[track.track_id])), (255,255,255), 2)
            '''
            cv2.rectangle(img, (int(bbox[0]+(area_Rate+(acc*6))*6*next_cen0[track.track_id]), int(bbox[1]+(area_Rate+(acc*6))*6*next_cen1[track.track_id])), (int(bbox[2]+(area_Rate+(acc*6))*6*next_cen0[track.track_id]),int(bbox[3]+(area_Rate+(acc*6))*6*next_cen1[track.track_id])), (255,255,255), 2)
            

            #cv2.line(img, (center[0],int(center[1]+h/2)),(center[0]+(6*next_cen0[track.track_id]),int(center[1]+h/2+(6*next_cen1[track.track_id]))), (0,0,0), 2)
            cv2.line(img, (center[0],int(center[1]+h/2)),(center[0]+int((area_Rate+(acc*6))*6*next_cen0[track.track_id]),int(center[1]+h/2+int((area_Rate+(acc*6))*6*next_cen1[track.track_id]))), (255,255,255), 2)
            
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                    continue
                #Adding a tail like by connecting the center using line with decreasing thickness
                thickness = int(np.sqrt(64/float(j+1))*2)
                
                cv2.line(img, (pts[track.track_id][j-1][0],int(pts[track.track_id][j-1][1]+h/2)), (pts[track.track_id][j][0],int(pts[track.track_id][j][1]+h/2)), color, thickness)

            #9. Counting the number of pedestrains(tracks) passing through and inside the barrier
            #Adding the barrier to image
            #cv2.line(img, (0, int(height/6+height/20)), (width, int(height/6+height/20)), (0, 255, 0), thickness=2)
            #cv2.line(img, (0, int(4*height/6-height/20)), (width, int(4*height/6-height/20)), (0, 255, 0), thickness=2)

            center_y = int(((bbox[1])+(bbox[3]))/2)

            #If the centroid of the track is inside the barrier, increase the count by 1
            if center_y >= int(height/6+height/20) and center_y <= int(4*height/6-height/20):
                if class_name == 'aeroplane':
                    counter.append(int(track.track_id))
                    current_count += 1

def final_display(img,output):
        #If the centroid of the track is Passing through the barrier, increase the count by 1
        total_count = len(set(counter))
        #Adding the current count and total count and other information to the image
        '''
        cv2.putText(img, "Pedestrain Count In Box Area: " + str(current_count), (0, 230), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Total Pedestrain Count: " + str(total_count), (0,280), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Format: aeroplane ID - Approaching(A)/Departing(D)/Left(L)/Right(R) - Time to Occlude", (0,30), 0, 1, (0,0,255), 2)
        cv2.putText(img, "- Apparent Speed(pixel/second(30 frames)) - Steps Walked", (0,80), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Occludee(1): Orange(Less Occlusion), Red(More Occulsion), Black(Missed In Previous Frame)", (0,130), 0, 1, (0,0,255), 2)
        cv2.putText(img, "New pedestrain/ aeroplane ID changed: White; No Occulsion(0): Green; Occluder(2): Blue", (0,180), 0, 1, (0,0,255), 2)
        '''

        #10. Calculating frames per second for processing each image  
        fps = 1./(time.time()-t1) #Number of frames processed(all modules) in a single second
        #cv2.putText(img, "FPS: {:.2f}".format(fps), (0,330), 0, 1, (0,0,255), 2) #Adding FPS to image
        out.write(img) #For storing the images as a video by appending
        img = imutils.resize(img, height=1080) #Making it 720*720
        cv2.imshow('output', img) #Dispalying the Cummulative Image
        #output = imutils.resize(output, height=1080)
        #cv2.imshow('output', output) #Dispalying the Cummulative Image
        
def little_things_matter(track,tracker,output):
            global color,bbox,center

            #for every track(bounding box) in the image
            class_name= track.get_class()

            #1. Missing aeroplane Detection in present frame And Reappearance In The Next Frame
            if(track.age>1 and track.time_since_update>1 and (class_name == 'aeroplane') and track.is_confirmed()):
                #If the track is not new and no new changes and class aeroplane and track is confirmed
                missed[track.track_id]=1 #Missing

            #2. Skipping of frame by a track
            if not track.is_confirmed() or track.time_since_update>1 or (class_name != 'aeroplane'):
                #If the track is not confirmed and track has no new changes and class not a aeroplane 
                return

            #8. Tracking the centroid like a tail(Trajectory)
            bbox =track.to_tlbr()
            center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2)) #Calculating center
            pts[track.track_id].append(center)



            depth[track.track_id]=0.5
            #16. Depth
            if(center[1]<vid_height and center[0]<vid_width):
                depth[track.track_id]=(center[1]+(track.to_tlwh()[3])/2)/vid_height
                print(depth[track.track_id])
                #depth[track.track_id]=output[center[1]][center[0]]

            area_Rate,acc=direction(track)
            color=Occlusion_rate(track,tracker,area_Rate,acc)
            velocity(track,area_Rate)



            #7. Identifying a new pedestrian/ aeroplane whose ID got changed 
            if track.age==1:
                color=(255,255,255) #Indicated by WHITE

            
            
            
            #14. Angle of pedestrain walk positive direction from x axis 
            height, width, _ = img.shape
            x=height/2
            y=width/2
            #print(center)
            '''
            if(center[0]>x and center[1]>y):
                print("4")
            elif(center[0]<x and center[1]<y):
                print("2")
            elif(center[0]<x and center[1]>y):
                print("1")
            else:
                print("3")
            '''

            #15. Time to collision
            timer=0
            t=0
            '''
            xlv=np.array([('A','AL'),('A','AR'),('D','DL'),('D','DR')])
            xc=np.array([('A','L'),('A','R'),('R','D'),('D','L'),('DL','DR'),('AR','AL')])
            cxxxv=np.array([('A','DL'),('A','DR'),('R','AL'),('R','DL'),('D','AL'),('D','AR'),('L','DR'),('L','AR')])
            opp=np.array([('L','R'),('AL','DR'),('AR','DL')])
            other=np.array([('A','D'),('R','AR'),('R','DR'),('L','DL'),('L','AL'),('AL','DL'),('DR','AR')])
            '''

            if(iou_ratio[track.track_id]>0):

                timer= (depth[cand_id]*np.linalg.norm(np.array(((area_Rate+(acc*6))*(6*next_cen0[track.track_id]),(area_Rate+(acc*6))*(6*next_cen1[track.track_id]))))+depth[track.track_id]*np.linalg.norm(np.array((6*next_cen0[cand_id],6*next_cen1[cand_id]))))/((depth[track.track_id]+depth[cand_id])*(speed[track.track_id]+speed[cand_id]))
                print(timer)
                '''
                if((dir[track.track_id] in opp[i,0] and dir[cand_id] in opp[i,1] for i in range(opp.shape[0])) or (dir[track.track_id] in opp[i,1] and dir[cand_id] in opp[i,0] for i in range(opp.shape[0]))):
                    print(np.array((next_cen0[track.track_id],next_cen1[track.track_id])))
                    timer= (depth[track.track_id]*np.linalg.norm(np.array((next_cen0[track.track_id],next_cen1[track.track_id])))+depth[cand_id]*np.linalg.norm(np.array((next_cen0[cand_id],next_cen1[cand_id]))))/((depth[track.track_id]+depth[cand_id])*(speed[track.track_id]+speed[cand_id]))
                elif((dir1[track.track_id] in xc[i,0] and dir1[cand_id] in xc[i,1] for i in range(xc.shape[0])) or (dir1[track.track_id] in xc[i,1] and dir1[cand_id] in xc[i,0] for i in range(xc.shape[0]))):
                    timer= (np.linalg.norm(np.array(next_cen0[track.track_id],next_cen1[track.track_id]))+np.linalg.norm(np.array(next_cen0[cand_id],next_cen1[cand_id])))/((speed[track.track_id]+speed[cand_id])/2)
                elif((dir1[track.track_id] in cxxxv[i,0] and dir1[cand_id] in cxxxv[i,1] for i in range(cxxxv.shape[0])) or (dir1[track.track_id] in cxxxv[i,1] and dir1[cand_id] in cxxxv[i,0] for i in range(cxxxv.shape[0]))):
                    timer= (np.linalg.norm(np.array(next_cen0[track.track_id],next_cen1[track.track_id]))+np.linalg.norm(np.array(next_cen0[cand_id],next_cen1[cand_id])))/((speed[track.track_id]+speed[cand_id])*0.75)
                elif((dir1[track.track_id] in xlv[i,0] and dir1[cand_id] in xlv[i,1] for i in range(xlv.shape[0])) or (dir1[track.track_id] in xlv[i,1] and dir1[cand_id] in xlv[i,0] for i in range(xlv.shape[0]))):
                    timer= (np.linalg.norm(np.array(next_cen0[track.track_id],next_cen1[track.track_id]))+np.linalg.norm(np.array(next_cen0[cand_id],next_cen1[cand_id])))/((speed[track.track_id]+speed[cand_id])*0.25)
                elif(dir1[track.track_id]==dir1[cand_id]):
                    timer= np.linalg.norm(np.array(pts[track.track_id][0])-np.array(pts[cand_id][0]))/(speed[track.track_id]-speed[cand_id])
                '''
                
            display(track,area_Rate,acc,timer)    

def preprocess():
    intialize()
    import core.utils as utils
    from core.yolov4 import filter_boxes
    from tensorflow.python.saved_model import tag_constants
    from core.config import cfg

    from deep_sort import preprocessing
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from tools import generate_detections as gdet
    global count,current_count,img,output
    count=0
    framework = 'tf'
    weights = './checkpoints/yolov4-tiny-416'
    size = 416
    tiny = True
    model = 'yolov4'
    iouvalue = 0.45
    score1 = 0.50
    dont_show = False
    info = False
    count1 = False
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector

    input_size = size

    # load tflite model if flag is set
    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']


    # get video ready to save locally if flag is set

    model_path1 = "Midas-Model-small.onnx"

    # Load the DNN model
    model1 = cv2.dnn.readNet(model_path1)

    if (model1.empty()):
        print("Could not load the neural net! - Check path")

    '''
    #Set backend and target to CUDA to use GPU
    model1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    '''


    frame_num = 0
    # while video is running
    while True:
        #Getting a next frame from the video
        _, frame = vid.read()
        count=count+1
        if frame is None:
            print('Completed')
            break
        if count%skip!=0:
            out.write(frame) #For storing the images as a video by appending
            continue
        img=frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_num +=1
        imgHeight, imgWidth, channels = frame.shape
        # Create Blob from Input Image
        # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        #blob = cv2.dnn.blobFromImage(img_in, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(frame, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)
        # Set input to the model
        model1.setInput(blob)

        # Make forward pass in model
        output = model1.forward()
        
        output = output[0,:,:]
        output = cv2.resize(output, (imgWidth, imgHeight))

        # Normalize the output
        output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        t1 = time.time() #start time

        # run detections on tflite if flag is set
        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if model == 'yolov3' and tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iouvalue,
            score_threshold=score1
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['aeroplane']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        cou = len(names)
        if count1:
            cv2.putText(frame, "Objects being tracked: {}".format(cou), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(cou))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]     
          

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        current_count = int(0) #Count of tracks(pedestrains) inside the barrier

        # update tracks
        for track in tracker.tracks:
            ids.append(track.track_id)
            little_things_matter(track,tracker,output)
        final_display(img,output)
        if cv2.waitKey(1) == ord('q'):
            #For exiting press q or automatically closes when video ends
            break
    
    vid.release() #Closes connection to input video file
    out.release() #Closes connection to output video file
    cv2.destroyAllWindows() #That we created

    if count>=30:
        for id in np.unique(ids):
            print(id)
            tol30(ptsGt[id],ptsExp30[id])
            tol60(ptsGt[id],ptsExp60[id])
            tol90(ptsGt[id],ptsExp90[id])
            tol120(ptsGt[id],ptsExp120[id])
            tol150(ptsGt[id],ptsExp150[id])
            tol180(ptsGt[id],ptsExp180[id])
    
    gc.collect()   

def graph(ptsG,ptsE,after):
    '''
    if(len(ptsG)>1):
        #plt.xlim(2*min(ptsGtX),2*max(ptsExpX))
        fig=plt.figure()
        #plt.ylim(500,1500)
        plt.title("Direction 3: GT vs predicted center of future position")
        plt.plot(ptsG[0], ptsG[1], label ='Ground truth')
        plt.plot(ptsE[0], ptsE[1], label ='Predicted center')
        plt.legend(loc ="upper right")
        plt.xlabel("Bbox Center's X coordinate")
        plt.ylabel("Bbox Center's Y coordinate")
        plt.savefig('./graphs dif/ACQUIRED/'+'A'+str(i)+'S'+str(j)+'D'+str(k)+'V'+str(l)+'after'+str(after)+'.png')
        #plt.figure().clear() 
        plt.close(fig)
    '''

    ptsE=[]


def tol30(ptsGt,ptsExp30):
    ptsG=[]
    ptsE=[]
    after=30
    toler=[]*0
    print(tolerance30)
    if len(ptsExp30)-((m-1)*mul)>0:
        for j in range(1, int(len(ptsExp30)-((m-1)*mul))):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[int(j+((m-1)*mul))])-np.array(ptsExp30[j])))
                    '''
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp30[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp30[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp30[j]))
                    '''
        if 3+((m-1)*mul)<len(ptsGt):
            if(len(toler)>0):
                tolerance30[k-1].append(np.mean(toler))
            tolerancex30[k-1].append(np.median(tolerx))
            tolerancey30[k-1].append(np.median(tolery))
            tolerancev30[k-1].append(np.var(toler))
        #print(tolerance30[k-1])
    #graph(ptsG,ptsE,after)
    ptsG=[]
    ptsE=[]
    
def tol60(ptsGt,ptsExp60):
    ptsG=[]
    ptsE=[]
    after=60
    toler=[]*0
    
    if len(ptsExp60)-((m-1)*mul)>0:
        for j in range(1, len(ptsExp60)-((m-1)*mul)):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[j+((m-1)*mul)])-np.array(ptsExp60[j])))
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp60[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp60[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp60[j]))
                    '''
                    print(wx,wy)
                    wx=wx+(lr*(ptsGt[j+((m-1)*mul)][0]-ptsExp60[j][0]))
                    wy=wy+(lr*(ptsGt[j+((m-1)*mul)][1]-ptsExp60[j][1]))     
                    print(wx,wy)
                    '''
        if 3+((m-1)*mul)<len(ptsGt):
            if(len(toler)>0):
                tolerance60[k-1].append(np.mean(toler))
            tolerancex60[k-1].append(np.median(tolerx))
            tolerancey60[k-1].append(np.median(tolery))
            tolerancev60[k-1].append(np.var(toler))
        #print(tolerance60[k-1])
    #graph(np.array(ptsG),np.array(ptsE),after)
    ptsG=[]
    ptsE=[]
    
def tol90(ptsGt,ptsExp90):
    mul=6
    ptsG=[]
    ptsE=[]
    after=90
    toler=[]*0

    if len(ptsExp90)-((m-1)*mul)>0:
        for j in range(1, len(ptsExp90)-((m-1)*mul)):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[j+((m-1)*mul)])-np.array(ptsExp90[j])))
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp90[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp90[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp90[j]))
        if 3+((m-1)*mul)<len(ptsGt):
            tolerance90[k-1].append(np.mean(toler))
            tolerancex90[k-1].append(np.median(tolerx))
            tolerancey90[k-1].append(np.median(tolery))
            tolerancev90[k-1].append(np.var(toler))
        #print(tolerance90[k-1])
    #graph(ptsG,ptsE,after)
    ptsG=[]
    ptsE=[]

def tol120(ptsGt,ptsExp120):
    mul=8
    ptsG=[]
    ptsE=[]
    after=120
    toler=[]*0

    if len(ptsExp120)-((m-1)*mul)>0:
        for j in range(1, len(ptsExp120)-((m-1)*mul)):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[j+((m-1)*mul)])-np.array(ptsExp120[j])))
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp120[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp120[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp120[j]))
        if 3+((m-1)*mul)<len(ptsGt):
            tolerance120[k-1].append(np.mean(toler))
            tolerancex120[k-1].append(np.median(tolerx))
            tolerancey120[k-1].append(np.median(tolery))
            tolerancev120[k-1].append(np.var(toler))
        #print(tolerance120[k-1])
    #graph(ptsG,ptsE,after)
    ptsG=[]
    ptsE=[]

def tol150(ptsGt,ptsExp150):
    mul=10
    ptsG=[]
    ptsE=[]
    after=150
    toler=[]*0
    
    if len(ptsExp150)-((m-1)*mul)>0:
        for j in range(1, len(ptsExp150)-((m-1)*mul)):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[j+((m-1)*mul)])-np.array(ptsExp150[j])))
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp150[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp150[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp150[j]))
                    
        if 3+((m-1)*mul)<len(ptsGt):
            tolerance150[k-1].append(np.mean(toler))
            tolerancex150[k-1].append(np.median(tolerx))
            tolerancey150[k-1].append(np.median(tolery))
            tolerancev150[k-1].append(np.var(toler))
        #print(tolerance150[k-1])
    #graph(ptsG,ptsE,after)
    ptsG=[]
    ptsE=[]

def tol180(ptsGt,ptsExp180):
    mul=12
    ptsG=[]
    ptsE=[]
    after=180
    toler=[]*0
    print(tolerance180)
    if len(ptsExp180)-((m-1)*mul)>0:
        for j in range(1, len(ptsExp180)-((m-1)*mul)):
            if j>3:
                if j+((m-1)*mul)<len(ptsGt):
                    toler.append(np.linalg.norm(np.array(ptsGt[j+((m-1)*mul)])-np.array(ptsExp180[j])))
                    tolerx.append(np.array(ptsGt[j+((m-1)*mul)][0])-np.array(ptsExp180[j][0]))
                    tolery.append(np.array(ptsGt[j+((m-1)*mul)][1])-np.array(ptsExp180[j][1]))
                    ptsG.append(np.array(ptsGt[j+((m-1)*mul)]))
                    ptsE.append(np.array(ptsExp180[j]))
        if 3+((m-1)*mul)<len(ptsGt):
            tolerance180[k-1].append(np.mean(toler))
            tolerancex180[k-1].append(np.median(tolerx))
            tolerancey180[k-1].append(np.median(tolery))
            tolerancev180[k-1].append(np.var(toler))
        #print(tolerance180[k-1])
    #graph(ptsG,ptsE,after)
    ptsG=[]
    ptsE=[]


tolerance30=[[]*0 for i in range(8)]
tolerance60=[[]*0 for i in range(8)]
tolerance90=[[]*0 for i in range(8)]
tolerance120=[[]*0 for i in range(8)]
tolerance150=[[]*0 for i in range(8)]
tolerance180=[[]*0 for i in range(8)] 

tolerancex30=[[]*0 for i in range(8)]
tolerancex60=[[]*0 for i in range(8)]
tolerancex90=[[]*0 for i in range(8)]
tolerancex120=[[]*0 for i in range(8)]
tolerancex150=[[]*0 for i in range(8)]
tolerancex180=[[]*0 for i in range(8)]

tolerancey30=[[]*0 for i in range(8)]
tolerancey60=[[]*0 for i in range(8)]
tolerancey90=[[]*0 for i in range(8)]
tolerancey120=[[]*0 for i in range(8)]
tolerancey150=[[]*0 for i in range(8)]
tolerancey180=[[]*0 for i in range(8)]

tolerancev30=[[]*0 for i in range(8)]
tolerancev60=[[]*0 for i in range(8)]
tolerancev90=[[]*0 for i in range(8)]
tolerancev120=[[]*0 for i in range(8)]
tolerancev150=[[]*0 for i in range(8)]
tolerancev180=[[]*0 for i in range(8)] 

k=2

'''
#ACQURIED
#1P
dirpd = [] #Direction of the pedestarin predicted
dirgt = [] #Direction of the pedestarin ground truth
for i in range(3,4):
    for j in range(1,7):
        for k in range(3,9):
            for l in range(1,2):
                print('C:/Users/91918/Desktop/DATABASE/1P/'+str(i)+'/S'+str(j)+'/D'+str(k)+'V'+str(l)+'.mp4')
                vid = cv2.VideoCapture('C:/Users/91918/Desktop/DATABASE/1P/'+str(i)+'/S'+str(j)+'/D'+str(k)+'V'+str(l)+'.mp4')
                _, img = vid.read()
                if img is None:
                    break
                codec = cv2.VideoWriter_fourcc(*'XVID')
                vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
                vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter('./data/video/ACQURIED/future/'+'A'+str(i)+'S'+str(j)+'D'+str(k)+'V'+str(l)+'.mp4', codec, vid_fps, (vid_width, vid_height))
                preprocess()
                
                cm=confusion_matrix(dirgt, dirpd)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print(cm)
                print(precision_score(dirgt, dirpd,average='micro'))
                print(recall_score(dirgt, dirpd,average='micro'))
                print(f1_score(dirgt, dirpd,average='micro'))
                
                gc.collect()
'''

'''
#TownCentre
k=2
for l in range(1,15):
            print("C:/Users/91918/Desktop/Datasets/cctv/Towncentre/part"+str(l)+"(split-video.com).mp4")
            vid = cv2.VideoCapture("C:/Users/91918/Desktop/Datasets/cctv/Towncentre/part"+str(l)+"(split-video.com).mp4")
            codec = cv2.VideoWriter_fourcc(*'XVID')
            vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
            vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('./data/video/TownCentre/'+str(l)+'.avi', codec, vid_fps, (vid_width, vid_height))
            preprocess()
            print(np.mean(tolerance30[1],axis=0))
            print(np.mean(tolerance60[1],axis=0))
            print(np.mean(tolerance90[1],axis=0))
            print(np.mean(tolerance120[1],axis=0))
            print(np.mean(tolerance150[1],axis=0))
            print(np.mean(tolerance180[1],axis=0))
            gc.collect()
'''
'''
import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))
x=[1,2,3,4,5,6,7,8]
y=[1,2,3,4,5,6,7,8]
sns.heatmap(cm, annot=True, fmt='.2f',xticklabels=x, yticklabels=y)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('cm3.png', bbox_inches='tight')
plt.show()
'''
'''
#2P
for i in range(3,4):
    for j in range(1,7):
        for k in range(1,9):
            for l in range(1,9):
                for m in range(1,6):
                    print('C:/Users/91918/Desktop/DATABASE/2P/'+str(i)+'/S'+str(j)+'/D'+str(k)+'D'+str(l)+'V'+str(m)+'.mp4')
                    vid = cv2.VideoCapture('C:/Users/91918/Desktop/DATABASE/2P/'+str(i)+'/S'+str(j)+'/D'+str(k)+'D'+str(l)+'V'+str(m)+'.mp4')
                    _, im = vid.read()
                    if im is None:
                        break
                    codec = cv2.VideoWriter_fourcc(*'XVID')
                    vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
                    vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out = cv2.VideoWriter('./data/video/ACQURIED/'+'A'+str(i)+'S'+str(j)+'D'+str(k)+'D'+str(l)+'V'+str(m)+'.mp4', codec, vid_fps, (vid_width, vid_height))
                    preprocess(vid,out)
                    #dirgt.append(k)
                    #dirpd.append(int(preprocess(vid,out,k)))
                    #print(confusion_matrix(dirgt, dirpd))
                    gc.collect()
'''

'''
#NITR
for j in range(1,22):
    for k in range(1,9):
        for l in range(1,4):
            print('C:/Users/91918/Desktop/NITRConsciousWalk_Part_1/10'+format(j,'02d')+'/10'+format(j,'02d')+'D'+str(k)+'S'+str(l)+'.avi')
            vid = cv2.VideoCapture('C:/Users/91918/Desktop/NITRConsciousWalk_Part_1/10'+str(j)+'/100'+str(j)+'D'+str(k)+'S'+str(l)+'.avi')
            codec = cv2.VideoWriter_fourcc(*'XVID')
            vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
            vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('./data/video/NITR/10'+format(j,'02d')+'D'+str(k)+'S'+str(l)+'.avi', codec, vid_fps, (vid_width, vid_height))
            preprocess()
            gc.collect()
'''
'''
#JAAD 

for l in range(251,347):
            print("C:/Users/91918/Desktop/Datasets/traffic/JAAD_clips/video_"+format(l,'04d')+".mp4")
            vid = cv2.VideoCapture("C:/Users/91918/Desktop/Datasets/traffic/JAAD_clips/video_"+format(l,'04d')+".mp4")
            codec = cv2.VideoWriter_fourcc(*'XVID')
            vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
            vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('./data/video/NITR/1001.avi', codec, vid_fps, (vid_width, vid_height))
            preprocess()
            gc.collect()
'''

vid = cv2.VideoCapture("SAM_C-RAM System in Action vs Fighter Jet - Surface-to-Air Missile - Military Simulation - ArmA 3.mp4")
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/TownCentreXVID.mp4', codec, vid_fps, (vid_width, vid_height))
preprocess()


for i in range(0,8):
    print(i)
    print(np.mean(tolerance30[i],axis=0))
    print(np.mean(tolerance60[i],axis=0))
    print(np.mean(tolerance90[i],axis=0))
    print(np.mean(tolerance120[i],axis=0))
    print(np.mean(tolerance150[i],axis=0))
    print(np.mean(tolerance180[i],axis=0))
for i in range(0,8):
    print(i)
    print(np.mean(tolerancex30[i],axis=0))
    print(np.mean(tolerancex60[i],axis=0))
    print(np.mean(tolerancex90[i],axis=0))
    print(np.mean(tolerancex120[i],axis=0))
    print(np.mean(tolerancex150[i],axis=0))
    print(np.mean(tolerancex180[i],axis=0))
for i in range(0,8):
    print(i)
    print(np.mean(tolerancey30[i],axis=0))
    print(np.mean(tolerancey60[i],axis=0))
    print(np.mean(tolerancey90[i],axis=0))
    print(np.mean(tolerancey120[i],axis=0))
    print(np.mean(tolerancey150[i],axis=0))
    print(np.mean(tolerancey180[i],axis=0))

for i in range(0,8):
    print(i)
    print(np.mean(tolerancev30[i],axis=0))
    print(np.mean(tolerancev60[i],axis=0))
    print(np.mean(tolerancev90[i],axis=0))
    print(np.mean(tolerancev120[i],axis=0))
    print(np.mean(tolerancev150[i],axis=0))
    print(np.mean(tolerancev180[i],axis=0))