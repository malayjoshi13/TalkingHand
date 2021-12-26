# TalkingHand
TalkingHand is a machine learning based conversation system which by help of trained weights of a **convolutional neural network** called as **VGG16 model** classifies and converts the hand gestures made by user into corresponding text based labels of those hand gestures. 

https://user-images.githubusercontent.com/71775151/120343446-5b191780-c316-11eb-8b19-4a9a685de2c3.mp4

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
**a)** Install Miniconda and Git Bash in your system. 

**b)** Use Git Bash to clone this repository in your system.

**c)** Then inside cloned repository, create a conda virtual environment and install required libraries using **environment.yml** file by following code in CMD
```
conda env create -f environment.yml
```
Then activate this virtual environment "TalkingHand",
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
<br>
<br>

### 2.3) Data Collection
**a)** Training and validation data corresponding to 7 types of hand gestures is collected using ```collection.py```.

**b)** A user can use ```collection.py``` file to create custom dataset by collecting data for label(s) of their choice. For example, data for label ```W``` can be collected by executing following command,
```
python collecting.py W
``` 

**c)** After executing the command, a window with a box on its right side will pop-up. There user is suppose to keep ```his/her hand``` and press ```key A, when ready to collect data``` (the same ```key A``` is also used to ```pause and resume image capturing process```). 

On doing this one more screen will appear on top of previously present screen. This screen will just show what kind of final images are getting saved on your disk as dataset. 

During this whole process, user can keep knowing the number of images captured till now from the count displayed on the top-left side of the first screen.

**Note**: While enacting hand gestures user need to constantly keep moving/waving their hand gesture as shown in testing video. The reason behind is that the hand gesture is thresholded from the background by making computer feed that what is moving/waving is hand gesture and what is been static is the background. Therefore user is required to ```keep movig/waving the hand gesture``` otherwise the algorithm will ```abruptly pause recording``` assuming that user is not present at that moment.

**d)** The recording ```stops``` either when the ```collecting.py``` script ```records 4000``` images of a particular label or user presses ```key Q``` on the keyboard. 

**e)** Once images corresponding to label W are collected, then user must go to the folder where those images are saved and there user must select which images to keep and which one to discard/delete. Once not so good images corresponding to label W are deleted, user is supposed to re-run again the same above command of ```python collecting.py W```, to capture more images to replace the deleted garbage images of label W. 

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

**b)** Then download ```training.ipynb``` python file from link https://drive.google.com/file/d/12QypAGmfiseZZ2r2vLktF-P_FYFAnwJg/view?usp=sharing, place it inside ```TalkingHand``` folder present in your Google Drive, open it and run each code cell.<br> 

**c)** After training, the weights file gets save in ```TalkingHand``` folder (present in Google Drive) which can be used like we did in section ```2.2```.<br> 
