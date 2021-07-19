# Object detection on Video using YOLO v3


### Project Information : 

Files you need for this Project or If you are using YOLO model :

1. file with extension .names : This file can be obtained from classes.txt file
2. train.txt and test.txt
3. classes.txt 
4 file with extension .data : This file can be obtained using train.txt and test.txt file

### Steps :


In YOLO first step is to draw bounding box around the target picture or video frames :
In case you are dealing with Video you need to convert the video into frames(Images) And To do that You can run this line of command in Terminal

***ffmpeg -i filename(it is videofile name) -vf fps=4 Image-%3d.jpg***

Here you can modify filename, fps and 3d, 2d or just d but It is better to keep 3D.

You will have frames (iMAGES) from the video. Now go to terminal again and 

**LabelImg** (Please make sure labelimage is installed on your computer)

Once this tool is opened you can select directory option for selecting the files. Select the folder which has frames(from video) and Use press character **W** to draw the bounding box around the object. Then save it.

Note : Make sure you have selected YOLO version to the saved file in the LabelImg tool. It has some other version by default for images to be saved in. hence change it to YOLO.

After saving this Bounding boxes drawn image in YOLO format. You will have two files 
1. Image_file_name.txt and  (This file will contain the X, Y coordinates of the objects in the Image)
2. Image_file_name.jpeg - ( This file contains the actual frames obtained from the video)
3. Classes.txt - It will contain the classes name (Object names which you have given)

These files should be in a Folder, In my case I have named this folder as Labelled Frames.

### First Step :

In this First step : 

We need to copy the full path of this folder. Either you can use the file in the repo to find the full path or simply get it from the computer itself . 
In order to get the full path from the code you can run this file ( **Get_Full_Path**) in the same folder (Labelled folder in my case).  Run this file from terminal. You will get the full path for this folder.  Whatever path you get . Just note it down somewhere for time being.

### Second step :

Here We will create train.txt and test.txt files - To do this. Run this python file **Creat_Train_Test_Txt_file.py** in the Repo.
This will create two files train.txt and test.txt . These two files will have full path of the folder having training Images and Testing Images.
And these two .txt files will be saved to the Folder where we have our label Frame (Bounding box images), In my case it is Labelled Image folder

### Third Step :

At this Step -

Will create files .data and .names

And To do that you need to run **Create_Data_Name_Extension_file.py**
Once We have these files. It will automatically be saved into the Label folder. In case not getting saved there. You move the files to one folder. In my case it is saved there in Label Frame folder


### Fourth Step :

At this step - 

You need to run the final file to detect the objects in the video clip-

python file name is **Video Object Detection.py or complete jupyter notebook file**





