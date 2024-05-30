import numpy as np
import cv2
import os
import imutils
import time

import time
from PIL import Image
import matplotlib.pyplot as plt


from _collections import deque
import gc
import statistics as st


NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2

def iou_1(bbox, candidates):
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

def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []
	final=[]


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, w, h), centroids[i])
			final.append([x, y, x+w, y+h, confidences[i]])
			results.append(res)            
	# return the list of results
	return results,final

def tlwh(bbox):
        bbox1=[0,0,0,0]
        bbox1[0:2]=bbox[0:2]
        bbox1[2] = bbox[2] - bbox[0]
        bbox1[3] = bbox[3] - bbox[1]
        return bbox1

def tlbr(bbox):
        bbox1=[0,0,0,0]
        bbox1[0:2]=bbox[0:2]
        bbox1[2] = bbox[0] + bbox[2]
        bbox1[3] = bbox[1] + bbox[3]
        return bbox1

def little_things_matter(box_id):
            global color,bbox,center,track_id

            #8. Tracking the centroid like a tail(Trajectory)
            bbox =box_id[0:4]
            track_id=int(box_id[4])
            center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2)) #Calculating center
            pts[track_id].append(center)



            depth[track_id]=0.5
            #16. Depth
            if(center[1]<vid_height and center[0]<vid_width):
                depth[track_id]=(center[1]+(tlwh(bbox)[3])/2)/vid_height
                #depth[track_id]=output[center[1]][center[0]]

            area_Rate,acc=direction(bbox,track_id)
            color,cand_id=Occlusion_rate(bbox,track_id,area_Rate,acc)
            velocity(area_Rate[track_id])


            #15. Time to collision
            timer=0
            t=0

            if(iou_ratio[track_id]>0):
                cand_id=int(cand_id)
                #cv2.imwrite(str(count)+".jpg",img)
                #timer= (depth[cand_id]*np.linalg.norm(np.array(((area_Rate[track_id])*(6*next_cen0[track_id]),(area_Rate[track_id])*(6*next_cen1[track_id]))))+depth[track_id]*np.linalg.norm(np.array(((area_Rate[cand_id])*6*next_cen0[cand_id],(area_Rate[cand_id])*6*next_cen1[cand_id]))))/((depth[track_id]+depth[cand_id])*(speed[track_id]+speed[cand_id]))
                timer= (np.linalg.norm(np.array(((area_Rate[track_id])*(6*next_cen0[track_id]),(area_Rate[track_id])*(6*next_cen1[track_id]))))+np.linalg.norm(np.array(((area_Rate[cand_id])*6*next_cen0[cand_id],(area_Rate[cand_id])*6*next_cen1[cand_id]))))/(speed[track_id]+speed[cand_id])


                
            display(color,bbox,area_Rate[track_id],acc,timer)  

def direction(bbox,track_id):
            global acc
            acc=0
            bbox = tlbr(bbox) #(min x, miny, max x, max y)
            
            #6. Detecting Approaching(A)/Departing(D)/Left(L)/Right(R)
            pres_X[track_id] = int(((bbox[0])+(bbox[2]))/2)
            #pres_Y[track_id] = int(((bbox[1])+(bbox[3]))/2)
            pres_Area[track_id]=(int(bbox[2])-int(bbox[0]))*(int(bbox[3])-int(bbox[1])) #Width*Height
            if(len(prev[track_id])>1 and len(prev_area_Rate[track_id])>1):
                area_Rate[track_id]=pres_Area[track_id]/np.mean(prev[track_id])
                #print(area_Rate[track_id])
                acc=area_Rate[track_id]-prev_area_Rate[track_id][0]
                acc=acc/6
                #print(acc)
                #area_Rate=pres_Area[track_id]/prev[track_id][0]
            if(area_Rate[track_id]>=1.075):
                #If the area of PRESENT bounding box is increasing relatively, Approaching
                if(pres_X[track_id]/prev_Avg_X[track_id]<=0.95):

                    #X Coordinate decreasing, going left
                    dir[track_id].append("6")
                elif(pres_X[track_id]/prev_Avg_X[track_id]>1.05):
                    #X Coordinate increasing, going right
                    dir[track_id].append("4")
                else:
                    dir[track_id].append("5")

            elif(area_Rate[track_id]<0.925):
                #If the area of PRESENT bounding box is decreasing relatively, Departing
                if(pres_X[track_id]/prev_Avg_X[track_id]<=0.95):
                    #X Coordinate decreasing, going left
                    dir[track_id].append("8")
                elif(pres_X[track_id]/prev_Avg_X[track_id]>1.05):
                    #X Coordinate increasing, going right
                    dir[track_id].append("2")
                else:
                    dir[track_id].append("1")

            else:
                #If the area of PRESENT bounding box is almost constant
                if(pres_X[track_id]/prev_Avg_X[track_id]<=1):
                    #X Coordinate decreasing, going left
                    dir[track_id].append("7")
                if(pres_X[track_id]/prev_Avg_X[track_id]>1):
                    #X Coordinate increasing, going right
                    dir[track_id].append("3")
            
            dir1[track_id]=st.mode(dir[track_id])
            
            
            prev[track_id].append((tlwh(bbox)[2])*(tlwh(bbox)[3]))
            
            prev_area_Rate[track_id].append(area_Rate[track_id])
            prev_X[track_id].append(pres_X[track_id])
            prev_Avg_X[track_id]=int(np.mean(prev_X[track_id])) #Average cummulative previous X
            return area_Rate,acc

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
    cen0 = [deque(maxlen=m) for i in range(50)] #center X for the pedestrian before 5 frames
    cen1 = [deque(maxlen=m) for i in range(50)] #center Y for the pedestrian before 5 frames
    next_cen0 = [0 for i in range(50)] #Sum of pairwise adjacent X center difference before 5 frames for predicting next bbox
    next_cen1 = [0 for i in range(50)] #Sum of pairwise adjacent Y center difference before 5 frames for predicting next bbox
    ratio = [0 for i in range(50)] #Occulsion IOU ratio with other neighbouring bounding boxes
    iou_ratio = [0 for i in range(50)] #Sum of Occulsion IOU ratio with other neighbouring bounding boxes
    missed = [0 for i in range(50)] #If the pedestrain goes missing for a frame and comes back in next frame
    dir = [deque(maxlen=30) for i in range(50)] #Direction of the pedestarin
    dir1 = [0 for i in range(50)] #Direction of the pedestarin
    depth = [0 for i in range(50)] #distance from camera
    toler=[]*0
    tolerx=[]*0
    tolery=[]*0
    ptsExp180 = [[]*0 for i in range(50)]
    ptsExp150 = [[]*0 for i in range(50)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp120 = [[]*0 for i in range(50)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp90 = [[]*0 for i in range(50)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp60 = [[]*0 for i in range(50)] #Centroids of each bounding boxes from atmost past 30 frames
    ptsExp30 = [[]*0 for i in range(50)]
    ptsGt = [[]*0 for i in range(50)]
    #prev_depth = [0 for i in range(10)] #previous cumulative distance from camera
    #prev_Avg_depth = [0 for i in range(10)] #Averege distance from camera
    #pres_Y = [0 for i in range(10)] #Center Y of present frame bounding box 
    #prev_Y = [0 for i in range(10)] #Center Y of previous frame bounding box 
    #prev_Avg_Y = [1 for i in range(10)] #Mean of Center Y of previous frames bounding box
    #prev_h = [0 for i in range(10)] #Center Y of previous frame bounding box 
    #prev_Avg_h = [1 for i in range(10)] #Mean of Center Y of previous frames bounding box
    #dirpt = [] #Direction of the pedestarin predicted
    prev = [deque(maxlen=18) for _ in range(50)] #Previous Areas of each bounding boxes from atmost past 30 frames
    area_Rate = [1 for _ in range(50)] #Previous Areas of each bounding boxes from atmost past 30 frames
    prev_area_Rate = [deque(maxlen=18) for _ in range(50)] #Previous Areas of each bounding boxes from atmost past 30 frames
    pres_Area = [0 for i in range(50)] #Area of present frame bounding box 
    prev_Area = [0 for i in range(50)] #Area of previous frame bounding box 
    prev_Avg_Area = [1 for i in range(50)] #Mean of areas of previous frames bounding box
    pres_X = [0 for i in range(50)] #Center X of present frame bounding box 
    prev_X = [deque(maxlen=18) for _ in range(50)] #Center X of previous frame bounding box 
    prev_Avg_X = [1 for i in range(50)] #Mean of Center X of previous frames bounding box
    steps = [0 for i in range(50)] #Steps walked so far
    speed = [0 for i in range(50)] #Speed of the pedestrain
    pts = [deque(maxlen=6) for _ in range(50)] #Centroids of each bounding boxes from atmost past 30 frames
    counter = [] #Counts number of pedestrains crossing the barrier

def Occlusion_rate(bbox,track_id,area_Rate,acc):
            global cand_id,Occulsion
            
            #11. Predecting the next bounding box for the pedestrian after 5 frames
            cen0[track_id].append(center[0]) #Appending center X points
            cen1[track_id].append(center[1]) #Appending center Y points


            if(len(abs(np.diff(cen0[track_id])))!=0):
                next_cen0[track_id]=int(mul*sum(np.diff(cen0[track_id]))) 
                next_cen1[track_id]=int(mul*sum(np.diff(cen1[track_id])))
            

            id=track_id
            #Selecting the neighbour candidates(tracks) bounding boxes of our PRESENT track if track is confirmed and has new changes and not present track and class person
            candidates = np.asarray([(tlwh(box_id[0:4])[0]+(area_Rate[int(box_id[4])])*(6*next_cen0[int(box_id[4])]), tlwh(box_id[0:4])[1]+(area_Rate[int(box_id[4])])*(6*next_cen1[int(box_id[4])]), tlwh(box_id[0:4])[2], tlwh(box_id[0:4])[3]) for box_id in boxes_ids if box_id[4]!=id])
            bbox = np.asarray((tlwh(bbox)[0]+(area_Rate[track_id])*(6*next_cen0[track_id]), tlwh(bbox)[1]+(area_Rate[track_id])*(6*next_cen1[track_id]), tlwh(bbox)[2], tlwh(bbox)[3])) #(top left x, top left y, width, height)

            if(len(candidates)!=0):
                area_intersection, area_candidates, area_bbox=iou_1(bbox,candidates)
                cand_id = np.asarray([box_id[4] for box_id in boxes_ids if box_id[4]!=id])
                
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
                        ratio[track_id]=area_intersection / (2*area_bbox - area_intersection)

                    #3.b For Relatively Bigger Bounding Boxes 
                    else:
                        #If our PRESENT track area is more than average of its neighbour candidates(tracks) areas
                        #We wont consider our bigger PRESENT track area for impartiality
                        ratio[track_id]=area_intersection / (2*area_candidates - area_intersection)
                    cand_id=cand_id[np.argmax(ratio[track_id])]
                    iou_ratio[track_id]=sum(ratio[track_id])

                else:

                    iou_ratio[track_id]=0
                
                
                color=(0,255,0)
                Occulsion=0
                #5. Detecting Occludee(1)/Occluder(2)
                if(len(area_candidates)!=0 and iou_ratio[track_id]>0.2):
                    
                    if(depth[track_id]<depth[int(cand_id)]):
                        #If our PRESENT track area is less than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is getting Occluded by atleast one of its neighbour candidates(tracks), so its an Occludee
           
                        color=(0,0,255)
                    else:
                        #If our PRESENT track area is more than minimum of its neighbour candidates(tracks) areas
                        #Our PRESENT track is Occluding atleast one of its neighbour candidates(tracks), so its an Occluder
                
                        color=(255,0,0) #Indicated by BLUE

                
            else:
                color=(0,255,0) #If no neighbour candidates(tracks), no occlusion, indiacted by GREEN
                Occulsion=0

            return color,cand_id

def velocity(area_Rate):
    #12. Apparent Speed of pedestrain per second(30 frames)
            speed[track_id]=0
            Euclidean=1
            
            for j in range(1, len(pts[track_id])):
                    #Euclidean distance of bounding box centers between adjacent frames
                    Euclidean=Euclidean+(np.linalg.norm(np.array(pts[track_id][j])-np.array(pts[track_id][j-1])))
            #speed[track_id]=(alpha*area_Rate+(1-alpha)*Euclidean)
            speed[track_id]=Euclidean
            
def display(color,bbox,area_Rate,acc,timer):

            h=tlwh(bbox)[3]
            #bbox=tlbr(bbox)
            #Adding the bounding box in the image
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            #Adding the pedestrain prodata/video/mixkit-people-in-the-subway-hall-in-tokyo-4454.mp4perties on top of bounding box in the image
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-90)), (int(bbox[0])+(len(str(track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-60)), (int(bbox[0])+(len(str(track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track_id))+7*len(str(Occulsion)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, str(track_id)+"-"+str(dir1[track_id])+"-"+str(round(depth[track_id],2)), (int(bbox[0]), int(bbox[1]-70)), 0, 1, (255, 255, 255), 2)
            cv2.putText(img, str(round(speed[track_id],1))+"-"+str(round(steps[track_id],1)), (int(bbox[0]), int(bbox[1]-40)), 0, 1, (255, 255, 255), 2)
            cv2.putText(img, str(round(timer,1)), (int(bbox[0]), int(bbox[1]-10)), 0, 1, (255, 255, 255), 2)
            
            
            cv2.rectangle(img, (int(bbox[0]+(area_Rate)*6*next_cen0[track_id]), int(bbox[1]+(area_Rate)*6*next_cen1[track_id])), (int(bbox[2]+(area_Rate)*6*next_cen0[track_id]),int(bbox[3]+(area_Rate)*6*next_cen1[track_id])), (255,255,255), 2)
            

            #cv2.line(img, (center[0],int(center[1]+h/2)),(center[0]+(6*next_cen0[track_id]),int(center[1]+h/2+(6*next_cen1[track_id]))), (0,0,0), 2)
            cv2.line(img, (center[0],int(center[1]+h/2)),(center[0]+int((area_Rate)*6*next_cen0[track_id]),int(center[1]+h/2+int((area_Rate)*6*next_cen1[track_id]))), (255,255,255), 2)
            
            for j in range(1, len(pts[track_id])):
                if pts[track_id][j-1] is None or pts[track_id][j] is None:
                    continue
                #Adding a tail like by connecting the center using line with decreasing thickness
                thickness = int(np.sqrt(64/float(j+1))*2)
                
                cv2.line(img, (pts[track_id][j-1][0],int(pts[track_id][j-1][1]+h/2)), (pts[track_id][j][0],int(pts[track_id][j][1]+h/2)), color, thickness)

            #9. Counting the number of pedestrains(tracks) passing through and inside the barrier
            #Adding the barrier to image
            #cv2.line(img, (0, int(height/6+height/20)), (width, int(height/6+height/20)), (0, 255, 0), thickness=2)
            #cv2.line(img, (0, int(4*height/6-height/20)), (width, int(4*height/6-height/20)), (0, 255, 0), thickness=2)

            center_y = int(((bbox[1])+(bbox[3]))/2)

            #If the centroid of the track is inside the barrier, increase the count by 1
            if center_y >= int(height/6+height/20) and center_y <= int(4*height/6-height/20):
                if class_name == 'person':
                    counter.append(int(track_id))
                    current_count += 1

def final_display(img):
        #If the centroid of the track is Passing through the barrier, increase the count by 1
        total_count = len(set(counter))
        #Adding the current count and total count and other information to the image
        '''
        cv2.putText(img, "Pedestrain Count In Box Area: " + str(current_count), (0, 230), 0, 1, (0, 0, 255), 2)
        cv2.putText(img, "Total Pedestrain Count: " + str(total_count), (0,280), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Format: Person ID - Approaching(A)/Departing(D)/Left(L)/Right(R) - Time to Occlude", (0,30), 0, 1, (0,0,255), 2)
        cv2.putText(img, "- Apparent Speed(pixel/second(30 frames)) - Steps Walked", (0,80), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Occludee(1): Orange(Less Occlusion), Red(More Occulsion), Black(Missed In Previous Frame)", (0,130), 0, 1, (0,0,255), 2)
        cv2.putText(img, "New pedestrain/ Person ID changed: White; No Occulsion(0): Green; Occluder(2): Blue", (0,180), 0, 1, (0,0,255), 2)
        '''

        #10. Calculating frames per second for processing each image  
        fps = 1./(time.time()-t1) #Number of frames processed(all modules) in a single second
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0,330), 0, 1, (0,0,255), 2) #Adding FPS to image
        out.write(img) #For storing the images as a video by appending
        img = imutils.resize(img, height=1080) #Making it 720*720
        cv2.imshow('output', img) #Dispalying the Cummulative Image
        #output = imutils.resize(output, height=1080);
        #cv2.imshow('output', output) #Dispalying the Cummulative Image

from sort import *
tracker = Sort()

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4.weights"
config_path = "yolov4.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

vid = cv2.VideoCapture("mixkit-people-in-the-subway-hall-in-tokyo-4454.mp4")
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('saved.mp4', codec, vid_fps, (vid_width, vid_height))

count=0
intialize()
while True:

    grabbed, img = vid.read()
    count=count+1
    if not grabbed:
        break
    if count%skip!=0:
         continue
    t1 = time.time() #start time
    results,final = pedestrian_detection(img, model, layer_name,personidz=LABELS.index("person"))
    if not final:
         continue
    boxes_ids = tracker.update(np.asarray(final))
    for box_id in boxes_ids:
        little_things_matter(box_id)
        final_display(img)

    if cv2.waitKey(1) == ord('q'):
        #For exiting press q or automatically closes when video ends
        break

vid.release()
cv2.destroyAllWindows()
        
  





