# ObjectDetection-using-YOLO
## Introduction
Yolo has gained a lot of popularity over the years since it is a real time detection method. This means that not only we will be able to classify an object but we will also locate it and extract the bounding box enclosing that object.<br>
Libraries used here are only OpenCV and numpy.


## Config and Weights
The cfg and the weight files can be downloaded from the official [yolo website](https://pjreddie.com/darknet/yolo/). The version recommend is YOLOv3-320.<br>
The coco names file can be downloaded [here](https://www.murtazahassan.com/wp-content/uploads/2020/06/cocoNames.zip]).
Since it was trained on the coco dateset we will first collect the names of our classes, this can be imported from the coco.names file which has the names of 80 classes.


## Loading the Model
Any model has two main components. One is the Architecture and the other is weights. For yolo3 we have separate files for both. So we will import the configuration file that has the architecture and the weights file that contains the weights.

```bash
config = 'yolov3.cfg'
weights = 'yolov3.weights'

net=cv2.dnn.readNetFromDarknet(config,weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```
## Input of image in Network
We cannot send our image form the camera directly to the network. It has to be in a certain format called blob. We can use the blobFromImage function which creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels. We will keep all the values at default.

```bash
blob=cv2.dnn.blobFromImage(img,1/255,(width,height),[0,0,0],1,crop=False)
net.setInput(blob)
```
## YOLO3 Architecture
![image](https://user-images.githubusercontent.com/51831819/115389428-b675c700-a1fa-11eb-95cf-4dcd0cad1f7c.png)<br>
To get the names we can use the getLayerNames fucnction. This returns all the names, but what we need are the names of only the output layers. So we can use the getUnconnectedOutLayers function which returns the indices of the output layers. Now we can simply use these indices to find the names from our layersNames list. Since we use 0 as the first element we have to subtract 1 from the indices we get form the getUnconnectedOutLayers function.


## Find Object
To Store the information of the relevant boxes we will create 3 lists. One would contain the information of the bounding box corner points, the other of class id with the highest confidence and last one with the confidence value of the highest class. Now after looping the 3 outputs we can get coordinates for the BoundingBox .As we know that each detection/box contains 85 values of which first 4 are cx,cy,w,h,Confidence and the rest 80 are class confidence values, we will remove the first 5 values from the detection. This will allow us to find the class index with the highest confidence values.<br>
Noo that we have the confidence value we can filter it. So we will add the confidence threshold. So if the confidence is grater than this, then only it will qualify as an object detected. Then we can get the pixel values of the x,y,w,h. To get pixel value we can simply multiply it with the width and height respectively.


## Non-Max Supression
To counter the problem of multiple bounding boxes around a single object we will use the Non Max Suppression. In the simplest term the NMS eliminates the overlapping boxes. It finds the overlapping boxes and then based on their confidence it will pick the max confidence box and supress all the non max boxes. So we will use the builtin NMSBoxes function. We have to input the bounding box points, their confidence values , the confidence threshold and the nmsThreshold. The function returns the indices after elimination.

```bash
indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold=0.4)
```

## Output
![Screenshot (25)](https://user-images.githubusercontent.com/51831819/115390316-ba561900-a1fb-11eb-9743-5b28a964a9ad.png)

The ouput was a bit laggy and accuracy was less which can be countered by using GPU with proper CUDA setups and using yolo.weights of model that was trained on high resolution images.

## Credits
This project wouldn't have been possible without https://www.murtazahassan.com/. So huge credit to him.

#### Hope you liked.

