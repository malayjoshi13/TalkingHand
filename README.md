# TalkingHand
TalkingHand is a machine learning based conversation system which by help of trained weights of a **convolutional neural network* called as *VGG16 model** classifies and converts the hand gestures made by user into corresponding text based labels of those hand gestures. 

https://user-images.githubusercontent.com/71775151/120343446-5b191780-c316-11eb-8b19-4a9a685de2c3.mp4

Users can further extend this system on their own set of hand gestures which will be discussed in the upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gestures made by user and then classify them to the corresponding labels. To do so, this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it. As a result it will understand different features of training images and then will map and memorize those features with their corresponding training labels.

![bandicam 2021-05-17 20-22-30-411](https://user-images.githubusercontent.com/71775151/118512165-30449600-b750-11eb-93e5-1a0724a8374c.jpg)

In this way the training process is carried out and the model layers of VGG16 + some additional layers are used together to memorize the mapping between features of images and their corresponding labels. Once done these memorizations are saved as in form of file called as weight file in hdf5 forma and then are used in future to classify a new and unseen scene consisting of hand gesture of user.
<br>
<br>

## 2) Usage

### 2.1) Setting up environment to use local system for testing and collecting data
**a)** Install Miniconda in your system from link:- https://docs.conda.io/en/latest/miniconda.html. 

Then, blindly keep agreeing to all prompts which come in the process. But be aware to tick mark both options of "Add Ananacaonda to my PATH environment variable" and "Register Anaconda as my default Python 3.7".

**b)** After this install Git Bash from link:- https://git-scm.com/downloads. 

After installing, open Git Bash and execute following code in it to clone my GitHub repository at any location of your wish in your local system. I choosed "c/Users/thispc/Desktop" location.

```
cd /c/Users/thispc/Desktop
```

Then execute following code,

```
git clone https://github.com/malayjoshi13/TalkingHand.git
```

**c)** Now open your command prompt (CMD) and type following code to change you current working directory/location to location of "TalkingHand" folder (name of the cloned repository in your local system)

```
cd C:\Users\thispc\Desktop\TalkingHand
```

Then in CMD, execute following code to create virtual environment named "TalkingHand" (as present at top of "environment.yml") and also install packages mentioned in the yml file,

```
conda env create -f environment.yml
```

After this, execute following code in CMD itself to activate the virtual environment "TalkingHand",

```
conda activate TalkingHand
```
With this you completed setup part.
<br>
<br>

### 2.2) Prediction
Now download weights from link https://drive.google.com/file/d/1-G0fSBWLO_W2w7OLWjPZTAHOsNKRGyLh/view?usp=sharing and place it inside TalkingHand folder present in your local system. After this execute following command in CMD,
```
python testing.py
```
On executing above command, a window will pop up where you can do hand gestures to execute following actions: <br>
a) printing character "A" <br>
b) printing character "B" <br>
c) printing character "C" <br>
d) deleting the character(s) <br>
e) creating space between two adjacent character(s) <br>
f) converting character(s) into audio <br>

Although current set of instructions are small in numbers but these can be further extended as per the need of user. The process to do so will be discussed in upcoming sections.
<br>
<br>

### 2.3) Data Collection
**2.3.1)** Training and validation data corresponding to 7 types of hand gestures is collected using ```collection.py``` python file to train and evaluate the model to get the ```weight``` which is used above for predicting the hand gestures of user.

One very ```interesting fact``` is that no matter initially on what training labels you have trained your model via ```training.py``` file, still during the prediction time via ```testing.py``` file you can change the name of labels according to your need.

In simple words, this can be explained like initially while training model using ```training.py``` my training labels were ```0```, ```1```, ```2```, ```3```, ```4```, ```5``` and ```nothing```. But later on while predicting labels for hand gestures, I changed the labels to ```A```, ```B```, ```C```, ```SPACE```, ```DELETE```, ```SPEAK``` and ```NOTHING```.

This was made possible because model do not take labels in form of name based labels but take and remember labels with a sequenced indexing. Thus during prediction time we simply changed the mapping between ordered indexing and previous label names to same previous ordered indexing and new label names.

Thus in this way data is collected for getting the weights which we used above.

**2.3.2)** Now let us understand how can we follow the same process and collect data for more such labels.

a) Before proceeding further make sure you have done step (1) of installing setup.

b) Now let us suppose user has to train model to classify hand gestures of alphabets ```W``` and ```Y```. In that case user will firstly collect data for label ```W``` by executing command,

```
python collecting.py W
``` 

After executing it a window will pop with a rectangular bounding box on right hand side. User is suppose to keep ```his/her hand within that box``` and ```when ready to collect data``` press ```key A``` on keyboard. 

On doing this one more screen will appear on top of previously present screen. This screen will just show what kind of final images are getting saved on your disk as dataset. User must shift this second screen away from screen 1 to make whole screen 1 visible. 

Also pressing key "A" will start capturing images of hand gestures enacted by user in front of webcamera by keeping hand within the bounding box. During this process user will be able to see the number of images captured till now on the left top side of the first window/screen.

**Note**: While enacting hand gestures user need to constantly keep moving/waving their hand gesture as shown in testing video. The reason behind is that the hand gesture is thresholded from the background by making computer feed that what is moving/waving is hand gesture and what is been static is the background. 

Therefore user is required to ```keep movig/waving the hand gesture``` otherwise the algorithm will ```abruptly pause recording``` assuming that user is not present at that moment. This might give a pause to recording. 

One way to ```pause``` the recording ```in genuine cases``` is to press key A and then to ```resume``` recording press ```key A again```.

The recording ```stops``` either when the ```collecting.py``` script ```records 4000``` images of a particular label or user press ```key Q``` on the keyboard. 

Once images corresponding to label W are collected, then user must go to the folder where those images are saved and there user must select which images to keep and which one to discard/delete. Once not so good images corresponding to label W are deleted, user is supposed to run again the same above command,

```
python collecting.py W
```

to capture more images to replace the deleted garbage images of label W. Repeat the process of capturing replacements and deleting the garbage images till user get very well defined 4000 images corresponding to label W.

***Note***: "Garbage in, garbage out" which means the images you feed model for training decides accuracy of trained model during predicting unknown and new images of similar context.

c) Once user will gather 4000 images for label W, he/she needs to perform same process with other label(s). Let's assume label Y. So firstly command:- python collecting.py Y, then placing hand in within the bounding box on the right side of popped up window, when ready pressing key A, moving the second popped up screen to a little more of left, keep moving hand while enactinggesture else a pause will occur, press keys to pause if genuinely needed, again key A to resume recording, doing hand gesture till the counter at left side of first screen reaches 4000 or tk quit in between by pressing Q (and again resumng can be done by command:- python collecting.py Y....the recording will start from place where previously left).

Once images are collected for label Y, go to the folder where the images are saved and there delete images which are not captured in correct manner. Once done re-execute the python file and collect more images, then again deleting and again capturing more images till all the 4000 images are well captured.

d) Once user gathers ```4000 images``` each for all the required labels (like we did for labels Y and W), then he/she has to execute following command in CMD:

```
python collecting.py final
```

Doing this will split 4000 images of each label into 80:20 ratio for training and validation purposes and will save in predefined manner inside folder named ```dataset_final``` which we will directly use during training.

e) This is how one of the collected images corresponding to label "2" looks like:-

![bandicam 2021-05-21 20-16-10-821](https://user-images.githubusercontent.com/71775151/119156053-79a72500-ba71-11eb-92ce-2bcaf2f97e5a.jpg)
<br>
<br>

### 2.4) Training on custom dataset
**a)** Make a folder in your own Google Drive named ```TalkingHand```. Inside this folder place ```final_dataset``` folder which will contain training and validation dataset.<br> 

**b)** Then download ```training.ipynb``` python file from link https://drive.google.com/file/d/12QypAGmfiseZZ2r2vLktF-P_FYFAnwJg/view?usp=sharing and place it inside ```TalkingHand``` folder present in your Google Drive.<br> 

**c)** Next open this ```training.ipynb``` file via Google Colaboratory application (will see the option at top of file) and execute each code cell.<br>  

**d)** After training, the weights file gets save in ```TalkingHand``` folder (present in Google Drive).<br> 
