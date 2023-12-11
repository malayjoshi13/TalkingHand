import time
import numpy as np
import operator
import cv2
import os
from keras.models import load_model

def removeBG(frame, bgModel, learningRate):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# First we load our trained model
model=load_model('weights.hdf5')

# parameters
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# initialise bg subtractor
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

# variables
isBgCaptured = 0   # bool, whether key "B" is pressed to capture the background
frame_count=0 # count the numbering of the frame opened via webcamera
previous_label_count = 0 # keep number of occurences of each predicted label
listt=list() # store all max frequencied predicted labels
word = ''
sentence = ''
start = 0 
text1=''
request = " "

# Now we specify that web-camera will be recording our images
cap = cv2.VideoCapture(0)

# Now we run an infinite loop
while True:
    # Next with following line we will read very first captured scene by our web camera and then store it in variable named 
    # "frame" 
    _, frame = cap.read()
    
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        
    # Then with next line we will swipe the recorded scene to make it look like mirror-view
    input = cv2.flip(frame, 1) 
    
    # Then we make the area of "frame" screen smaller to only capture hand gesture
    smaller_region = input[80:450, 350:640]
    if isBgCaptured==0:
        cv2.imshow('Background Capture', smaller_region)
    
    # Then we create a rectangular region on "smaller_region" screen for performing hand gestures
    # Note: coordinates are in format of top left coordinate (side, top) and bottom right coordinate (side, top)
    # origin at top left
    cv2.rectangle(smaller_region, (15,15),(270,310), (255, 255, 255), 2)    
    
    # Then under the hood we make copy of area inside that rectangular region and store it in "roi" named variable
    # This will be used to predict the corresponding label
    roi = smaller_region[15:310, 15:270] 
    
    roi = removeBG(roi, bgModel, learningRate)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    _, mask = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    if isBgCaptured==0:
        cv2.imshow('Mask', mask)

    # Then we resize this extracted area into the size required by trained model to do prediction
    resized_roi = cv2.resize(mask, (64, 64)) 

    if isBgCaptured == 1:
        # Right now shape of "mask" is (64, 64). To let it get predicted using trained model we have to make it 
        # of shape (1, 64, 64, 3). For that first we use "merge" function
        channeled_mask = cv2.merge((resized_roi,resized_roi,resized_roi))
        # Then we use "reshape" function
        reshaped_mask = channeled_mask.reshape(1, 64, 64, 3)
        
        # And finally predict the probability score of labels corresponding to each scene captured by the web camera
        # i.e. let us assume that "reshape_mask" has hand gesture corresponding to one of the 7 training labels called "one"
        # ( which by virtue of sorting is placed at second position in ordering when we see result of "(generator.class_indices)" ) 
        # then variable "result" will have value of [[0. 1. 0. 0. 0. 0. 0.]]
        result = model.predict(reshaped_mask) 
        
        # Then we create a dictonary to map probability position/indexing with label
        prediction_dictionary ={'A': result[0][0], 
                    'B': result[0][1], 
                    'C': result[0][2],
                    'SPACE': result[0][4],
                    'DELETE': result[0][3],
                    'D': result[0][5]
                    }                         
        
        # Next code line you are about to see do sorting, find label with max probability and place that label-probability pair at the first position 

        # How??
        # To understand it let us first see the following example: 

        # prediction_dictionary = {'a': 2, 'b': 4, 'c': 3, 'd': 1, 'e': 0}
        # prediction = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        # print(prediction)
        # >> [('b', 4), ('c', 3), ('a', 2), ('d', 1), ('e', 0)]

        # So in above example what is happening is that firstly using code "prediction_dictionary.items()", pairs like ('b', 4) are picked up 
        # and then using code "operator.itemgetter(1)" these pairs are sorted according to second parameter of like 4, 3, 2,..... (and not according to 
        # first parameter like b, c, a,...)
        # the second parameter of whichever pair will be highest will be placed at first (descending order) position of variable "prediction" by use of 
        # "reverse=True" i.e. as "4" is highest of all other alike parameters thus the pair of "('b', 4)" is placed at first position

        # The same thing happens with below code line also
        # And on basis of what we are assuming till now, prediction = [('A', 1.0), ('B', 0.0), ('C', 0.0), ('SPACE', 0.0), ('DELETE', 0.0), 
        # ('CLEAR', 0.0), ('NOTHING', 0.0), ]
        prediction = sorted(prediction_dictionary.items(), key=operator.itemgetter(1), reverse=True)  

        # Next "prediction[0][0]" will pick up element at first position of variable "prediction"
        top_label=prediction[0][0]  
        listt.append(top_label)     
        
        #################################################

        # Till "listt" is not more than 20 items, we need to tell "final_label" aka winning label.
        # To do this, we first see if its "NOTHING" or not, using non-zero pixels.
        # If it is not "NOTHING" we pick each element of "listt" (here elements refers to labels with max probabilities in every iteration) 
        # and count occurence of each element and store it in variable "current_label_count".        
        if len(listt) < 20:  

            # Now using background extracted filtered image we got from above, we calculate number of non-zero/white pixels. 
            # If this value is less than "1" then we say that the filtered image has no gesture and thus we classify it as "NOTHING"   
            non_zero_pixels = (cv2.countNonZero(resized_roi)*100)/(4096)
            print(non_zero_pixels)

            if(non_zero_pixels<1):           
                listt.clear()
                previous_label_count=0
                final_label='NOTHING'


            else:
                # After this, the count in variable "current_label_count" is compared with count stored in "previous_label_count" (which is initially "0") and
                # if occurence of label which is currently been picked up is more than the occurenece of prev label, then we say the current label is more accurate prediction
                # as it has occured more/ predicted more number of times. This label is stored as "final_label".
                for i in listt:
                    current_label_count=listt.count(i)
                    if(current_label_count>previous_label_count):
                        previous_label_count=current_label_count
                        final_label=i 

        # As "listt" has more than 20 elements, to avoid any wrong label dominate whole time, we erase the whole list and starts a new list.
        # Doing this gives chance that if at first model has predicted the gesture wrongly then in the new list right label would have chance to be famous 
        else:
            listt.clear()
            previous_label_count = 0

        #################################################
        
        # If the favourable predicted labels are either "A", "B" or "C", then we add these words to the existing text called as "sentence" displayed on screen 
        # by doing "sentence = sentence + word"  

        # Here we have used the variables "previous_label_count" and "start". They will prevent the same word to get print on screen multiple
        # times in a single go. Same (or any another character also) character can only be printed after 3s.
        
        # How? It will be implemented by making "previous_label_count" variable as "1" which was earlier "0". Doing this will prevent mutiple printing of same character on screen as the 
        # next character could only be added to existing variable "sentence" (and later displayed on screen) only and only when variable "count" is zero (which at present is 1).

        # Later on when predicted label would be "NOTHING" then clock will be stopped and time will be measured. If time passed is more than 3s only then 
        # "previous_label_count" variable will become "0" and then we could add same character again or even any other character also. 
                
        if final_label == 'A':
            if previous_label_count == 0:
                sentence = sentence + 'A' 
                previous_label_count = 1 
            if previous_label_count == 1:            
                start = time.time()
                request = "wait before adding next character...."

        # Same logic as used for label "A" 
        if final_label == 'B':
            if previous_label_count == 0:
                sentence = sentence + 'B'
                previous_label_count = 1
            if previous_label_count == 1:            
                start = time.time() 
                request = "wait before adding next character...."
                
        # Same logic as used for label "A" 
        if final_label == 'C':
            if previous_label_count == 0:
                sentence = sentence + 'C'
                previous_label_count = 1
            if previous_label_count == 1:
                start = time.time() 
                request = "wait before adding next character...."

        # Same logic as used for label "A" 
        if final_label == 'D':
            if previous_label_count == 0:
                sentence = sentence + 'D'
                previous_label_count = 1
            if previous_label_count == 1:
                start = time.time() 
                request = "wait before adding next character...."

        # Same logic as used for label "A" 
        if final_label == 'SPACE':
            if previous_label_count == 0:
                sentence = sentence + ' '
                previous_label_count = 1
            if previous_label_count == 1:
                start = time.time() 
                request = "wait before adding next character...."       
                
        # Same logic as used for label "A" and a twist that here we one-by-one delete each character from existing text "sentence" been displayed on screen. 
        # How? We do this by ignoring the last character of string "sentence" and printing rest whole string. Then in next execution
        # we again miss the last character of string to print 
        if final_label == 'DELETE':
            if previous_label_count==0: 
                sentence = sentence[:(len(sentence)-1)] 
                previous_label_count = 1
            if previous_label_count == 1:            
                start = time.time()
                request = "wait before adding next character...."
                
        # If the predicted label is "NOTHING", then we.....
        if final_label == 'NOTHING':
                #....firstly measure the current time and then find difference between current time and time which we measured above and stored in "start" variable.
                # This difference is stored in variable "interval" and if its greater than "3", i.e. "3s" then we make variable "count" as "0". Doing this will let user
                # to again print same character or any other character after printing the previous character. 
                end = time.time()
                interval = end-start
                
                if interval>3:
                    request = " "
                    previous_label_count = 0
                    request = "PLEASE DO YOUR HAND GESTURE"    
                    
                # This code will help to start a new line to add characters of variable "sentence" by deleting the completely
                # filled string called as "sentence"
                elif len(sentence)<35:
                    sentence = sentence
                else:
                    text1="REACHED LIMIT, DELETING OUTPUT"
                    sentence = ''
                    text1 = ''
        
        #################################################

        # Now we will create two screens using "output" and "smaller_region" and merge them to a single cv window called as
        # "output".
        
        # Then comes second screen which is made up of the frame captured above by web-camera, which we refered as "input". 
        # On this screen we display "final_label" variable.
        # Where "final_label" will have name of label which is most popularly predicted most mumber of times 
        cv2.putText(smaller_region, final_label, (70, 350), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 6) #it picks first element from above series, which is actually biggest prob label   
        

        # First screen will be formed using a black screen made from "np.zeroes" and called as "output". 
        # On "output" we put three texts: "request", "sentence" and "text1".
        # Where "request" will display either blank string or "PLEASE START YOUR HAND GESTURES" or 
        # "wait before adding next character...." according to situations,
        # "text1" prints "REACHED LIMIT, DELETING OUTPUT" after "sentence" has full limit 
        # and "sentence" prints the final sentence to be printed
        output = np.zeros((370, 1000, 3), dtype=np.uint8)
        cv2.putText(output, request, (42, 90), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 4)
        cv2.putText(output, sentence, (44, 180), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 255), 4)
        cv2.putText(output, text1, (20, 270), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 0, 255), 4)

 
        # Then we merge both screens horizontally
        merging_horizontally = np.concatenate((smaller_region, output),axis=1)
        # and display as cv window named "output"
        
        # Then with above merged screens we merge an image horizontally
        instructions = cv2.imread('./static/instructions.png')
        instructions = cv2.resize(instructions, (1290, 392))
        merging_vertically = np.concatenate((merging_horizontally, instructions),axis=0)
        # And display as cv window
        cv2.imshow("output", merging_vertically)
       
        #################################################


    k = cv2.waitKey(10)
    if k == ord('q'): # quit the console --> key Q on keyboard
        break
    elif k == ord('b'): # press 'b' to capture the background
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'): # press 'r' to reset the background
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')  

cap.release()
cv2.destroyAllWindows()
