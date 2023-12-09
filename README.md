# TalkingHand
TalkingHand is a Computer Vision and Deep Learning-based **Sign Language to Text conversion system** which with the help of fine-tuned **convolutional neural network** of **VGG16**, classifies and converts the hand gestures made by the user into corresponding text-based labels. 

https://user-images.githubusercontent.com/71775151/120343446-5b191780-c316-11eb-8b19-4a9a685de2c3.mp4

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) Getting Started

### 1.1) To set up the environment on the local system for **"inference"** and **"collecting custom dataset"**, run the following command:

```
git clone https://github.com/malayjoshi13/TalkingHand.git

cd TalkingHand

conda env create -f environment.yml

conda activate TalkingHand
```

### 1.2) Inference (Hand Gesture to Text)

Now download weights from (this link)[https://drive.google.com/file/d/1-G0fSBWLO_W2w7OLWjPZTAHOsNKRGyLh/view?usp=sharing] and place it inside the "TalkingHand" folder present in your local system. After this, run the following command:

```
python scripts/gesture_to_text.py
```

On executing the above command, a window will pop up where you can make hand gestures to execute the following actions: <br>
 
![instructions](https://user-images.githubusercontent.com/71775151/147411597-b9ce18f7-ef47-48a4-8e8b-0dc10e626610.jpg)

a) printing character "A" <br>
b) printing character "B" <br>
c) printing character "C" <br>
d) deleting the character(s) <br>
e) creating space between two adjacent character(s) <br>
f) converting character(s) into audio <be>

### 1.3) Collecting custom dataset for re-training
**a)** Training and validation data corresponding to 7 types of hand gestures is collected using the ```collecting.py``` script. For example, data for label ```W``` can be collected by executing the following command,

```
python scripts/collecting.py W
``` 

**b)** After executing the command, a window with a box on its right side will appear. There user is supposed to keep ```his/her hand``` and press ```key A when ready to collect data``` (the same ```key A``` is also used to ```pause and resume image image-capturing process```). 

On doing this, one more screen will appear showing what kind of final images are getting saved on the user's disk as a dataset. 

During this whole process, the user can keep knowing the number of images captured till now from the count displayed on the top-left side of the first screen.

**Note**: The user needs to constantly keep moving/waving their hand gesture while enacting hand gestures, as shown in the testing video. The reason is that the hand gesture is thresholded from the background by making the computer feel that what is moving/waving is a hand gesture and what has been static is the background. Therefore the user is required to ```keep movig/waving the hand gesture``` otherwise, the algorithm will ```abruptly pause recording``` assuming that the user is not present.

**c)** The recording ```stops``` either when the ```collecting.py``` script ```records 4000``` images of a particular label or user presses ```key Q``` on the keyboard. 

**d)** Once images corresponding to label W are collected, the user must go to the folder where those images are saved, and the user must select which images to keep and which ones to discard/delete. Once not-so-good images corresponding to label W are deleted, the user should re-run the above command of ```python collecting.py W```to capture more images to replace the deleted garbage images of label W. 

**g)** Once the user gathers ```4000 images``` each for all the required labels (like we did for one label of "W"), then he/she has to execute the following command:

```
python scripts/collecting.py final
```

This command will split those collected 4000 images of each label into 80:20 ratio for training and validation purposes and then save those images in a  predefined manner inside the ```dataset_final``` folder. This folder will be used during the training process.

### 1.3) Re-training on custom dataset

After collecting the custom dataset locally, run this script in Google Colab, [training.ipynb](https://github.com/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) and follow each instruction in that script. 

This script will clone this Github repo, ask you (the user) to upload the custom dataset (located in the local system) and train the underlying model. After training, the weights file gets saved in the ```TalkingHand``` folder (present in Google Drive), and then we can use it locally for inference (exactly like we did in section 1.2).
