
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2 as cv
import random
import pickle
import sys
import logging
import time
import datetime
import pyscreenshot as ImageGrab
import ctypes
import pyopencl as cl

# (1) setup OpenCL
platforms = cl.get_platforms() # a platform corresponds to a driver (e.g. AMD)
platform = platforms[0] # take first platform
devices = platform.get_devices(cl.device_type.GPU) # get GPU devices of selected platform
device = devices[0] # take first GPU
context = cl.Context([device]) # put selected GPU into context object
queue = cl.CommandQueue(context, device) # create command queue for selected GPU and context

print("Platform: {} Device:{}".format(platform, device))

# Parse the screen size
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Initialize the parameters
confThreshold = 0.20  #Confidence threshold
nmsThreshold = 0.40   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
classes = None

# Capture a frame from screen
def getFrame():
    # create a bounding box at the center of the screen
    bbox=((screensize[0]/2)-208, (screensize[1]/2)-208, (screensize[0]/2)+208, (screensize[1]/2)+208)
    # grab screen
    img = ImageGrab.grab(bbox)
    # save image file
    img.save('screenshot.png')
    # convert image to numpy array
    img_np = np.array(img)
    print(type(img_np))
    # convert color space from BGR to RGB
    cap = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
    print(type(cap))
    return cap

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    #print(layersNames)
    #print([layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
    global classes
    # if classId == ''
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
     
    label = '%.2f' % conf
         
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
 
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left+2, bottom-2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    global confThreshold  #Confidence threshold
    print(confThreshold)
    global nmsThreshold   #Non-maximum suppression threshold
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    print("out = {}".format(outs))
    for out in outs:
        for detection in out:
            #print("detection : {}".format(detection))
            scores = detection[5:]
            #print("scores len : {}".format(len(scores)))
            classId = np.argmax(scores)
            confidence = scores[classId]
            #print(confidence)
            if confidence > confThreshold:
                print("Object detected")
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        print("Left:{} Top:{} Width:{} Height:{}".format(left, top, width, height))
        if width > 410:
            width = 410
        if height > 410:
            height = 410
        if left < 5:
            left = 5
        if top < 5:
            top = 5
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

def main():
    global classes
    global inpWidth
    global inpHeight
    # Load names of classes
    classesFile = "coco.names";
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    print(classes)
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3-tiny.cfg";
    modelWeights = "yolov3-tiny.weights";
 
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
    
    while cv.waitKey(0):
        # Get a frame to process
        frame = getFrame()
        #cv.imshow("Frame", frame)
        #cv.waitKey()
        print("Outputing image to file")
        outputFile = 'screenshot' + '_yolo_out_py.jpg'

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        # a horizontal strip with 3 one-channel images.
        #blobb = blob.reshape(blob.shape[2] * blob.shape[1], blob.shape[3], 1) 
        #cv.imshow("Blob", blobb)
        #cv.waitKey()

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
 
        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)
 
        # Put efficiency information. The function getPerfProfile returns the 
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
 
        # Write the frame with the detection boxes
        #cv.imwrite(outputFile, frame.astype(np.uint8));
        
        # Live show detection
        cv.imshow("Frame", frame)

if __name__ == '__main__':
    main()