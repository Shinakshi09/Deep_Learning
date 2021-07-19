#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2


""" Loading the cascade for the face """
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

""" Loading the cascade for the eyes """
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[5]:


""" Creating a function that takes as input the image in black and white (gray) and the original image (frame), 
     and that will return the same image with the detector rectangles"""

def detect(gray, frame):
    
    
    """ Applying the detectMultiScale method from the face cascade to locate one or several faces in the image."""
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    
    """ For each detected face """
    for (x, y, w, h) in faces:
        
        """ Paint a rectangle around the face """
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        """ Get the region of interest in the black and white image."""
        roi_gray = gray[y:y+h, x:x+w] 
        
        """ Get the region of interest in the colored image """
        roi_color = frame[y:y+h, x:x+w]
        
        """ Apply the detectMultiScale method to locate one or several eyes in the image """
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        """ For each detected eye """
        for (ex, ey, ew, eh) in eyes:
            
            """ Paint a rectangle around the eyes, but inside the referential of the face."""
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
            
    """ Return the image with the detector rectangles."""
    return frame 


# In[7]:


""" To turn the Camera On """

video_capture = cv2.VideoCapture(-1) # -1 for linux machine


""" Repeat infinitely (until break) """
while True:
    
    """ Get the last frame """
    _, frame = video_capture.read() 
    
    """ Let's do some colour transformations."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    """ Get the output of our detect function."""
    canvas = detect(gray, frame)
    
    """ Display the output """
    cv2.imshow('Video', canvas)
    
    """ Type q To stop the loop"""
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
""" Turn the camera off"""
video_capture.release()

""" TO destroy all the windows inside which the images were displayed """
cv2.destroyAllWindows() 


# In[ ]:




