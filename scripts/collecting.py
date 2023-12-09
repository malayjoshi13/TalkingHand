#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing modules which we require
import cv2
import os
import sys
import shutil 

# When user code "python collecting.py label", then "sys.argv[1]" takes that "label" and save it in "label_name" variable
label_name = sys.argv[1]

# Then we pick up the location of current working directory and save it in variable "location"
location=os.getcwd()

# Then we set some parameters. First one is for number of images for each label. And second one is for images for training purpose.
num_samples = 4000
training_images_number = 3200

# So if label in "python collecting.py label" is "final", i.e. user has collected 4000 images for his/her labels and now want to split them to 80:20 ratio for training and testing purpose then execute 
# commands inside this loop.
if label_name=="final":
  # First we check that user has first collected images and has not directly jumped to splitting function
    if not os.path.exists(location+"/"+"dataset"):
        print("before using splitter, please gather some data corresponding to some labels")
    # If user has collected images and then has come to use splitting function then we let him/her to execute following code to do 80:20 splitting into training and testing datasets.
    else: 
        # Inside this loop we pick each folder of each label.
        for folders in os.listdir(location+"/"+"dataset"):
            counter=0
            # Check that all those labels' folders have 4000 images been collected
            if len(os.listdir(location+"/"+"dataset"+"/"+folders))!=num_samples:
                print("some label has less than 4000 images, please collect more")
                break
            # If 4000 images are collected of each label then we execute following code
            else:    
                # It will create two folders "train" and "validation" inside folder "final_dataset". This folder will be then used in training process.
                if not os.path.exists(location+"/"+"final_dataset"+"/"+"train"+"/"+folders):
                    os.makedirs(location+"/"+"final_dataset"+"/"+"train"+"/"+folders)
                if not os.path.exists(location+"/"+"final_dataset"+"/"+"validation"+"/"+folders):
                    os.makedirs(location+"/"+"final_dataset"+"/"+"validation"+"/"+folders)

                # Next we move 3200 images of a particular label into "train" folder and rest 800 images into "validation" folder. In short we do 80:20 split of total data of each label 
                # for training and testing purpose. 
                for files in os.listdir(location+"/"+"dataset"+"/"+folders):
                    counter+=1
                    if counter<training_images_number:
                        source=location+"/"+"dataset"+"/"+folders+"/"+files
                        destination=location+"/"+"final_dataset"+"/"+"train"+"/"+folders+"/"+files
                        dest = shutil.move(source, destination) 
                    else: 
                        source=location+"/"+"dataset"+"/"+folders+"/"+files
                        destination=location+"/"+"final_dataset"+"/"+"validation"+"/"+folders+"/"+files
                        dest = shutil.move(source, destination)                       

# Now let us discuss case when label in "python collecting.py label" is not "final", i.e. user is still in process of collecting images for some labels (whose names be like zero, one, A, etc).                   
else:   
    # In such case we first make folders for all those labels inside a main folder named "dataset" and according to situations we specify values of variable "count" which stores number of collected images
    # for a particular label
    directory = location+"/"+"dataset"+"/"+label_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        count = 0
    else:
        print("already exists") 
        count=len(os.listdir(directory))

        # Then below is a small code that will help to rename all files in a serial order of 1.jpg, 2.jpg....to 4000.jpg. 
        # Actually after user collects images and then deletes those which are not properly captured then in between values get miss due to file deletion. To get back that serial order of saving file's name
        # in a serial order starting from "1" and ending at "4000".
        i = 50000
        for filename in os.listdir(directory):
          new = directory + "/" + str(i) + ".jpg"
          prev = directory + "/" + filename
          os.rename(prev, new)
          i += 1   

        i = 1
        for filename in os.listdir(directory):
          new = directory + "/" + str(i) + ".jpg"
          prev = directory + "/" + filename
          os.rename(prev, new)
          i += 1 
    
    # Now we specify that web-camera will be recording our images
    cap = cv2.VideoCapture(0)

    # This variable will be used later to either pause or resume recording based on fact that it is set to "True" or "False"
    start = False

    # Then we initialise background subtraction algorithm which will be used later to seperate our moving hand gestures from the static background
    subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(2,False,20,True)

    # Now we run an infinite loop
    while True:

        # And record the frames from web-camera and store it to "frame" variable
        ret, frame = cap.read()

        # Then we flip the captred frame to mirror view
        frame = cv2.flip(frame, 1)

        # Now if user captures all 4000 images then web-camera will itself stops due to this line
        if count == num_samples:
            break   

        # Then we draw a bounding box on the captured frame for user to keep hand gestures within that box
        cv2.rectangle(frame, (300,65),(635,415), (255, 255, 255), 2)

        # Now if the variable "start" which we defined above is set to "True" (by user pressing key "A" once), then we will enter into this loop
        if start:
            # Firstly we will capture whatever is present inside the bounding box and save it to variable "roi"
            roi = frame[65:415, 300:635]
            # Next we will resize it to let our trained model (which was trained on training images of 64) to work upon this captured image.
            roi = cv2.resize(roi, (64, 64))  
            # Then we apply the background subtraction algorithm which we defined above. This line will extract user's moving hand from the static background and will threshold hand pixels with white pixels
            # while background pixels as black. It saves this image as "mask".
            mask = subtractor.apply(roi, -1)
            # We see this image "mask" through cv2 window named "Mask"
            cv2.imshow("Mask", mask)
            # Then we count number of white and black pixels in the filtered image "mask". Based on this count we get a upper hand to classify whether the image "mask" has some hand gesture (more of white
            # pixels) or no hand gesture (more of non-white pixels) is detected within the bounding box. If there are more of white pixels then we save that image assuming there would be hand gesture in 
            # that frame, otherwise we skip the frame from saving.
            p = (cv2.countNonZero(mask)*100)/(4096)
            if p>5:     
                cv2.imwrite(directory+"/"+str(count + 1)+'.jpg', mask)
                count += 1
            else:
                continue

        # Next we put count of number of images captured on the top left corner of the frame and show this frame as a cv2 window named "Colecting images"           
        cv2.putText(frame, "Collecting {}".format(count), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Collecting images", frame)

        # Next we give user option to:     
        k = cv2.waitKey(10)
        # start/pause/resume --> key A on keyboard
        if k == ord('a'):  
            start = not start
        # quit the console --> key Q on keyboard    
        if k == ord('q'):  
            break    
                  
    cap.release()
    cv2.destroyAllWindows()

