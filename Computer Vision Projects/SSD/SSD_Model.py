#!/usr/bin/env python
# coding: utf-8

# In[11]:


""" Importing the libraries """
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

import warnings
warnings.filterwarnings("ignore")


# ## Defining a function that will do the detections 

# In[13]:





""" About this function :
We define a detect function that will take as inputs, a frame, a ssd neural network, 
and a transformation to be applied on the images, 
and that will return the frame with the detector rectangle"""

def detect(frame, net, transform): 
    
    """ height and the width of the frame """
    height, width = frame.shape[:2]
    
    """ Applying the transformation to our frame."""
    
    frame_t = transform(frame)[0]
    
    """ Converting the frame into a torch tensor """
    
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    
    """ Add a fake dimension corresponding to the batch """
    x = tensor(x.unsqueeze(0))
    
    """ Feeding the neural network ssd with the image and we get the output y."""
    y = net(x)
    
    """ Creating the detections tensor contained in the output y"""
    detections = y.data
    
    """ create a tensor object of dimensions [width, height, width, height """
    scale = torch.Tensor([width, height, width, height])
    
    
    """ For every class """
    for i in range(detections.size(1)):
        """ Initialize the loop variable j that will correspond to the occurrences of the class"""
        j = 0
        
        """ Take into account all the occurrences j of the class i that have a matching score larger than 0.6."""
        while detections[0, i, j, 0] >= 0.6:
            
            """ Get the coordinates of the points at the upper left and the lower right of the detector rectangle."""
            pt = (detections[0, i, j, 1:] * scale).numpy()
            
            """ Drawing a rectangle around the detected object."""
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            
            """ put the label of the class right above the rectangle."""
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            """ Increment j to get to the next occurrence."""
            j += 1
            
            """ Return the original frame with the detector rectangle and the label around the detected object."""
    return frame


# ## Creating the SSD neural network

# In[14]:





"""  We create an object that is our neural network ssd."""
net = build_ssd('test')

""" Weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth)."""
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))


# ## Creating the transformation

# In[15]:


"""Creating the transformation"""

"""   Creating an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network."""
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))


# ## Object Detection from the video

# In[16]:


""" Open/Read the video """

reader = imageio.get_reader('funny_dog.mp4')

""" Let's take the fps frequence (frames per second)."""
fps = reader.get_meta_data()['fps']

""" create an output video with this same fps frequence """
writer = imageio.get_writer('output.mp4', fps = fps)

""" Will Iterate on the frames of the output video:"""

for i, frame in enumerate(reader):
    
    """ Calling our detect function (defined above) to detect the object on the frame."""
    frame = detect(frame, net.eval(), transform)
    
    """ Adding the next frame in the output video. """
    writer.append_data(frame)
    
    """ Printing the number of the processed frame."""
    print(i)
    
    """ Closing the process that handles the creation of the output video"""
writer.close()


# In[ ]:




