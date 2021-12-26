# TalkingHand
TalkingHand is a machine learning based conversation system which by help of trained weights of a **convolutional neural network* called as *VGG16 model** classifies and converts the hand gestures made by user into corresponding text based labels of those hand gestures. 

https://user-images.githubusercontent.com/71775151/120343446-5b191780-c316-11eb-8b19-4a9a685de2c3.mp4

Users can further extend this system on their own set of hand gestures which will be discussed in the upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.
<br>
<br>

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gestures made by user and then classify them to the corresponding labels. To do so, this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it. As a result it will understand different features of training images and then will map and memorize those features with their corresponding training labels.

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
Now download weights from link https://drive.google.com/file/d/1-G0fSBWLO_W2w7OLWjPZTAHOsNKRGyLh/view?usp=sharing and place it inside "TalkingHand" folder present in your local system. After this execute following command in CMD,
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
**a)** Training and validation data corresponding to 7 types of hand gestures is collected using ```collection.py```.

**b)** A user can use the same script to make his/her own custom dataset by collecting data for label(s) of their choice. For example, data for label ```W``` can be collected by executing following command,

```
python collecting.py W
``` 

**c)** After executing the command, a window with a box on its right side will pop-up. There user is suppose to keep ```his/her hand``` and press ```key A, when ready to collect data```. The same ```key A``` is also used to ```pause and resume image capturing process```. 

On doing this one more screen will appear on top of previously present screen. This screen will just show what kind of final images are getting saved on your disk as dataset. Thus, shift this second screen away from firstly presented screen so that only firstly presented screen would be visible. 

During this whole process, user from the count displayed on the top-left side of the first screen will keep knowing the number of images captured till now.

**Note**: While enacting hand gestures user need to constantly keep moving/waving their hand gesture as shown in testing video. The reason behind is that the hand gesture is thresholded from the background by making computer feed that what is moving/waving is hand gesture and what is been static is the background. Therefore user is required to ```keep movig/waving the hand gesture``` otherwise the algorithm will ```abruptly pause recording``` assuming that user is not present at that moment.

**d)** The recording ```stops``` either when the ```collecting.py``` script ```records 4000``` images of a particular label or user presses ```key Q``` on the keyboard. 

**e)** Once images corresponding to label W are collected, then user must go to the folder where those images are saved and there user must select which images to keep and which one to discard/delete. Once not so good images corresponding to label W are deleted, user is supposed to run again the same above command of ```python collecting.py W```, to capture more images to replace the deleted garbage images of label W. 

**f)** User needs to repeat this same process of capturing images and deleting the garbage images till user get very well defined 4000 images corresponding to label W.

Example: This is how one of the collected images corresponding to label "2" looks like:-

![bandicam 2021-05-21 20-16-10-821](https://user-images.githubusercontent.com/71775151/119156053-79a72500-ba71-11eb-92ce-2bcaf2f97e5a.jpg)

**g)** Once user gathers ```4000 images``` each for all the required labels (like we did for label "W"), then he/she has to execute following command in CMD:

```
python collecting.py final
```

to split those 4000 images of each label into 80:20 ratio for training and validation purposes and then to save those images in a  predefined manner inside ```dataset_final``` folder. This folder will going to be used during training process.
<br>
<br>

### 2.4) Training on custom dataset
**a)** Make a folder in your own Google Drive named ```TalkingHand```. Inside this folder place ```final_dataset``` folder which will contain training and validation dataset.<br> 

**b)** Then download ```training.ipynb``` python file from link https://drive.google.com/file/d/12QypAGmfiseZZ2r2vLktF-P_FYFAnwJg/view?usp=sharing and place it inside ```TalkingHand``` folder present in your Google Drive.<br> 

**c)** Next open this ```training.ipynb``` file via Google Colaboratory application (will see the option at top of file) and execute each code cell.<br>  

**d)** After training, the weights file gets save in ```TalkingHand``` folder (present in Google Drive).<br> 
