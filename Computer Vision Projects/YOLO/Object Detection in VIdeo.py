#!/usr/bin/env python
# coding: utf-8

# In[2]:


""" 

Steps :


1. Reading Image

2. Getting BLog

3. Loading YOLO v3 Network

4. Implementing forward pass

5. Getting bounding Boxes

6. No maximum supression - We filter the bounding boxes using this technique called NO Maximum supression

7. Drawing bounding boxes with labels


and As a Result. Will see openCV window with original image and bounding boxes with labels around detected Objects

"""

import numpy as np
import cv2
import time


video = cv2.VideoCapture('forest-road.mp4')

"""

Creating variable for handling the frames of the video , Assigning it to None because i do not want to initialize it for now

without knowing the height and width of the video frame


"""
writer = None

"""

Variable for spatial dimensions of the frames, same doing here too

"""
h, w = None, None





"""

Loading YOLO v3 network
"""

with open('classes.names') as f:
    
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolov3.cfg',
                                     'yolov3.weights')


layers_names_all = network.getLayerNames()


layers_names_output =     [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


"""

Setting minimum probability to eliminate the weak predictions

"""

probability_minimum = 0.5

"""

Let's color our bounding boxes around the detected objects

"""


threshold = 0.3

"""


Now Will read the frames in the Loop


Defining variable for counting frames

At the end we will show total amount of processed frames

"""

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


f = 0


t = 0

while True:
    # Capturing frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved
    # e.g.: at the end of the video,
    # then we break the loop
    if not ret:
        break

    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)


# In[6]:


"""

Implementing Forward pass

Implementing forward pass with our blob and only through output layers
Calculating at the same time, needed time for forward pass


Setting blog as input to the neural network

"""
network.setInput(blob) 
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

"""

Increasing the counters for total frames

"""
f += 1
t += end - start

"""

Time spent on single frame

"""
print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))


"""

Getting bounding boxes

Preparing lists for detected bounding boxes,
obtained confidences and class's number

"""
bounding_boxes = []
confidences = []
class_numbers = []

"""
Going through all output layers after feed forward pass
    
"""
for result in output_from_network:
    
        
    """
    Going through all detections from current output layer
    """
    for detected_objects in result:
        
            
        scores = detected_objects[5:]
        """
        Getting index of the class with the maximum value of probability
         
        """
        class_current = np.argmax(scores)
        """
        Getting value of probability for defined class
            
        """
        confidence_current = scores[class_current]

        """
        Eliminating weak predictions with minimum probability
            
        """
        if confidence_current > probability_minimum:
            
            """
                Scaling bounding box coordinates to the initial frame size
                YOLO data format keeps coordinates for center of bounding box
                and its current width and height
                That is why we can just multiply them elementwise
                to the width and height
                of the original frame and in this way get coordinates for center
                of bounding box, its width and height for original frame
                
            """
                
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            """
                
                Now, from YOLO data format, we can get top left corner coordinates
                that are x_min and y_min
                
            """ 
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            """ 
                
                Adding results into prepared lists
                
            """ 
            bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)


# In[11]:


"""
Non-maximum suppression

Implementing non-maximum suppression of given bounding boxes
With this technique we exclude some of bounding boxes if their
corresponding confidences are low or there is another
bounding box for this region with higher confidence
    
"""
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)


"""

Drawing bounding boxes and labels

Checking if there is at least one detected object
after non-maximum suppression

"""
if len(results) > 0:
    for i in results.flatten():
        """ 
        Getting current bounding box coordinates,
        its width and height
        """
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        """
        Drawing bounding box on the original current frame
        
        """
        cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

        """
        Preparing text with label and confidence for current bounding box
        """
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

        """
        Putting text with label and confidence on the original image
        
        """
        cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)


"""
 Writing processed frame into the file

"""

if writer is None:
    
        
    """ 
        Constructing code of the codec
        to be used in the function VideoWriter
        
    """
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    """ 
        
        Writing current processed frame into the video file
        Write processed current frame to the file
    """
    writer.write(frame)



"""
FInal results
"""
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# In[ ]:



video.release()
writer.release()

