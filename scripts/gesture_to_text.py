import time
import numpy as np
import operator
import cv2
import sys, os
from IPython.display import clear_output
from keras.models import load_model
from gtts import gTTS
from playsound import playsound

# First we load our trained model
model=load_model('weights.hdf5')

# Then we start webcamera
cap = cv2.VideoCapture(0)

# Next we use a variable which will be used later to count the numbering of the frame opened via webcamera
frame_count=0

# After this we use a variable which will be used later to keep number of occurences of each predicted label 
previous_label_count = 0

# Then we initiate a list which will be used later to store all max frequencied predicted labels
listt=list()

# Next we initiate a filter that will be used later to seperate out our moving hand from the static background 
subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(2,False,20,True)

# After this we'll perform following jobs decribed inside this loop till our web camera will be ON
while(cap.isOpened()):

    # Now using variable "frame_count" which we declared above,
    # we will initialise all variables to be used later as soon as web camera captures very first scene.
    # After first scene we will not re-initialise the variables
    frame_count+=1  
    if frame_count==1:
        word = ''
        sentence = ''
        start = 0 
        text1=''
        request = " "
    
    # Next with following line we will read very first captured scene by our web camera and then store it in variable named 
    # "frame" 
    _, frame = cap.read()
    
    # Then with next line we will swipe the recorded scene to make it look like mirror-view
    input = cv2.flip(frame, 1) 
    
    # Then we make the area of "frame" screen smaller to only capture hand gesture
    smaller_region = input[80:450, 350:640]
    
    # Then we create a rectangular region for performing hand gestures
    # Note: coordinates are in format of top left coordinate (side, top) and bottom right coordinate (side, top)
    # origin at top left
    cv2.rectangle(smaller_region, (15,15),(270,310), (255, 255, 255), 2)    
    
    # Then under the hood we make copy of area inside that rectangular region and store it in "roi" named variable
    # This will be used to predict the corresponding label
    roi = smaller_region[15:310, 15:270] 
    
    # Then we resize this extracted area into the size required by trained model to do prediction
    resized_roi = cv2.resize(roi, (64, 64)) 
    
    # Next we convert the resized copy into black-white format same as that of training image. 
    # Will do it by using a kind of filter initiated above which will help us to seperate our moving hand from the 
    # static background.
    mask = subtractor.apply(resized_roi, -1)  

    # Now using background extracted filtered image we got from above, we calculate number of non-zero/white pixels 
    # in it and store the calculated data in variable called as "non_zero_pixels". 
    # If this value is less than "1" then we say that the filtered image has no gesture and thus we classify it as "NOTHING"   
    non_zero_pixels = (cv2.countNonZero(mask)*100)/(4096)
    if(non_zero_pixels<1):           
        listt.clear()
        previous_label_count=0
        final_label='NOTHING'
    
    # Right now shape of "mask" is (64, 64). To let it get predicted using trained model we have to make it 
    # of shape (1, 64, 64, 3). For that first we use "merge" function
    channeled_mask = cv2.merge((mask,mask,mask))
    # Then we use "reshape" function
    reshaped_mask = channeled_mask.reshape(1, 64, 64, 3)
    
    # And finally predict the probability score of labels corresponding to each scene captured by the web camera
    # i.e. let us assume that "reshape_mask" has hand gesture corresponding to one of the 7 training labels called "one"
    # ( which by virtue of sorting is placed at second position in ordering when we see result of "(generator.class_indices)" ) 
    # then variable "result" will have value of [[0. 1. 0. 0. 0. 0. 0.]]
    result = model.predict(reshaped_mask) 
    
    # Then we create a dictonary to map probability position/indexing with label
    # i.e. we will map first probebility result in list "result" to the label "A"...this means from now onwards probability number placed at 
    # first position of list "result" which is "result[0][0]" will now correspond to the probability of model predicting label "A"

    # In same we map other variables with other probability indexes

    # Advantage of this dictionary is that we can either continue with index-label mapping as followed during the training process and could be seen 
    # by "generator.class_indices" like "result[0][0]"--> label "ZERO"  ............or..............  we can change the mapping like we didi here
    # "result[0][0]"--> label "A"
    prediction_dictionary ={'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'SPACE': result[0][3],
                  'DELETE': result[0][4],
                  'SPEAK': result[0][5],
                 'NOTHING': result[0][6]
                  }                         
    
    # This line do sorting, find label with max probability and place that label-probability pair at the first position 

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
    # i.e. it will pick up second parameter of the pair with max probability and placed at first position in variable "prediction"
    # i.e. based on assumption we are taking right now, a = A
    top_label=prediction[0][0]  
    # All of these first-first labels (corresponding to max probabilities of every prediction round) gets stored in list called "listt"
    listt.append(top_label)     
    
    # To avoid any wrong label to rule whole time, what we do is that once "listt" has 20 elements we erase the whole list and starts a new list.
    # Doing this gives more chance that if at first iteration model has predicted the gesture wrongly then in the new list right label would have 
    # chance to be famous           
    if len(listt) < 20:   
        # After creating function that will clear the list named "listt" after it get full by 20 elements, next thing is to pick each element of
        # that list (here elements refers to labels with max probabilities in every iteration) and to count occurence of each element and store
        # it in variable "current_label_count". 

        # After this, the count in variable "current_label_count" is compared with count stored in "previous_label_count" (which is initially "0") and
        # if former is greater than latter, i.e. if occurence of label which is currently been picked up is more than the occurenece of label
        # evaluated before it, then we say the current label is more accurate prediction as has occured more/ predicted more number of times.
        # Then finally whichever label has occured max time in whole evaluation process is considered most favourable prediction and we store it in
        # variable "final_label".
        for i in listt:
            current_label_count=listt.count(i)
            if(current_label_count>previous_label_count):
                previous_label_count=current_label_count
                final_label=i 
    else:
        listt.clear()
        previous_label_count = 0
    
    
    # If the favourable predicted labels are either "A", "B" or "C", then we add these words to the existing text called as "sentence" displayed on screen 
    # by doing "sentence = sentence + word"  

    # Here we have used concept of variables "count" and "start". What these variables will do is that they will prevent the same word to get print on screen multiple
    # times in a single go. Same (or any another character also) character can only be printed after 3s.
    
    # How? It will be implemented by making "count" variable as "1" which was earlier "0". Doing this will prevent mutiple printing of same character on screen as the 
    # next character could only be added to existing variable "sentence" (and later displayed on screen) only and only when variable "count" is zero (which at present is 1).

    # So to let user print same character again or even any other character also (by making "count" variable as "0"), we initiated clock and stores the time in variable 
    # "start". Later on when predicted label would be "NOTHING" then this clock will be stopped and time will be measured. If time passed is more than 3s only then 
    # "count" variable will become "0" and then we could add same character again or even any other character also. 
    if final_label == 'A':
        if count == 0:
            sentence = sentence + 'A' 
            count = 1 
        if count == 1:            
            start = time.time()
            request = "wait before adding next character...."

    # Same logic as used for label "A" but with a twist that here we add character "B" between already present character and nect upcoming character
    if final_label == 'B':
        if count == 0:
            sentence = sentence + 'B'
            count = 1
        if count == 1:            
            start = time.time() 
            request = "wait before adding next character...."
            
    # Same logic as used for label "A" but with a twist that here we add character "C" between already present character and nect upcoming character
    if final_label == 'C':
        if count == 0:
            sentence = sentence + 'C'
            count = 1
        if count == 1:
            start = time.time() 
            request = "wait before adding next character...."
            final_label="wait.."
            
    # Same logic of "count" and "start" variables as used for label "A" but with a twist that here we add a "space" between already present character and 
    # next upcoming character
    if final_label == 'SPACE':
        if count == 0:
            sentence = sentence + ' '
            count = 1
        if count == 1:
            start = time.time() 
            request = "wait before adding next character...."
            
    # adding the functionality to convert text into audio        
    if final_label == 'SPEAK':
        if count == 0 and len(sentence)!=0:
            language = 'en'
            audio = gTTS(text=sentence, lang=language, slow=False)
            audio.save("speech.mp3")
            playsound("speech.mp3")
            os.remove("speech.mp3")
            count = 1
        elif count == 0 and len(sentence)==0:
            continue
        if count == 1:
            start = time.time() 
            request = "wait speech is going on...."            
            
    # Same logic of "count" and "start" variables as used for label "A" but with a twist that here we one-by-one delete each character from existing text "sentence"
    # been displayed on screen. How? We do this by ignoring the last character of string "sentence" and printing rest whole string. Then in next execution
    # we again miss the last character of string to print 
    if final_label == 'DELETE':
        if count==0: 
            sentence = sentence[:(len(sentence)-1)] 
            count = 1
        if count == 1:            
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
                count = 0
                request = "PLEASE DO YOUR HAND GESTURE"    
                
            # This code will help to start a new line to add characters of variable "sentence" by deleting the completely
            # filled string called as "sentence"
            if len(sentence)<35:
                sentence = sentence
            else:
                text1="REACHED LIMIT, DELETING OUTPUT"
                sentence = ''
                text1 = ''
     

    # Now we will create two screens using "output" and "smaller_region" and merge them to a single cv window called as
    # "output".
    
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

    # Then comes second screen which is made up of the frame captured above by web-camera, which we refered as "input". 
    # On this screen we display "final_label" variable.
    # Where "final_label" will have name of label which is most popularly predicted most mumber of times 
    cv2.putText(smaller_region, final_label, (70, 350), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,0), 6) #it picks first element from above series, which is actually biggest prob label   
    
    # Then we merge both screens horizontally
    merging_horizontally = np.concatenate((smaller_region, output),axis=1)
    # and display as cv window named "output"
    
    # Then with above merged screens we merge an image horizontally
    instructions = cv2.imread('instructions.jpg')
    instructions = cv2.resize(instructions, (1290, 392))
    merging_vertically = np.concatenate((merging_horizontally, instructions),axis=0)
    # And display as cv window
    cv2.imshow("output", merging_vertically)
    
    # Now if user want to exit he/she must press "Q" key on their keyboard
    interrupt = cv2.waitKey(10)
    if interrupt == ord('q'): 
        break
  
cap.release()
cv2.destroyAllWindows()

