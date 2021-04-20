import cv2
import numpy as np

cap=cv2.VideoCapture(0)

# creating a list of classnames from coco.names
classnames=[]
classfile='coco.names'
with open(classfile,'rt') as f:
    classnames=f.read().rstrip('\n').split('\n')

config = 'yolov3.cfg'
weights = 'yolov3.weights'

net=cv2.dnn.readNetFromDarknet(config,weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

width=320
height=320
confidencethreshold=0.5

def find_object(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confidence=[]
    for output in outputs:
        for det in output:
            # making a scores array to store [cx. cy, w, h, %confidence]
            scores=det[5:]
            # get Id that have maximum %confidence of 80 classes
            classId=np.argmax(scores)
            confi=scores[classId]
            if confi>confidencethreshold:
                # to get width and height of bbox
                w,h=int(det[2]*wT), int(det[3]*hT)
                # to get x, y of bbox since cx, cy are the centres we have to subtract w/2 and h/2 respectively
                x,y=int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confidence.append(float(confi))
    indices=cv2.dnn.NMSBoxes(bbox,confidence,confidencethreshold,nms_threshold=0.2)

    # To draw bounding boxes and writing classnames on it
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{classnames[classIds[i]].upper()} {int(confidence[i]*100)}%',(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        
while True:
    _,img= cap.read()

    # We have to use the blobFromImage function which creates 4-D blob from image.
    blob=cv2.dnn.blobFromImage(img,1/255,(width,height),[0,0,0],1,crop=False)
    net.setInput(blob)

    # To find out all the laeyrs present in the darknet model
    layernames=net.getLayerNames()

    # we have to get only the UnconnectedLayers of the CNN model
    outputnames=[(layernames[i[0]-1]) for i in net.getUnconnectedOutLayers()]

    # forward pass to get outputs
    outputs=net.forward(outputnames)

    # This outputs array has 3 more arrays
    # the first is of size 300x85 where 300 is the number of boundingboxes
    # 85 includes(cx, cy, width, height, %confidence, 80 classes having their respective %confidence)
    # the second has 1200x85 and third has 4800x85

    find_object(outputs,img)
    cv2.imshow("output",img)
    cv2.waitKey(1)

