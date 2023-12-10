# TalkingHand
TalkingHand is a Computer Vision and Deep Learning-based **Sign Language to Text conversion system** which with the help of fine-tuned **convolutional neural network** of **VGG16**, classifies and converts the hand gestures made by the user into corresponding text-based labels. Custom dataset of about 4000 images each for 6 labels has been collected for fine-tuning the CNN model using a combination of ```createBackgroundSubtractorMOG2``` and ```color threshold``` techniques so that data collected will have a lower bias due to the shape & colour of user's hand making the gesture and altering lightning conditions.

https://user-images.githubusercontent.com/71775151/120343446-5b191780-c316-11eb-8b19-4a9a685de2c3.mp4

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) Getting Started

### 1.1) To set up the environment on the local system for **"inference"** and **"collecting custom dataset"**, run the following command:

```
git clone https://github.com/malayjoshi13/TalkingHand.git

cd TalkingHand

conda create -n talkinghand_env

conda activate talkinghand_env

pip install -r requirements.txt
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

**b)** After executing the command, a window with a box on its right side will appear. First user has to press ```key B``` (ensure Capslock to be OFF) to capture background. During this time user should not plca hand within blue bounding box. After this user is supposed to keep his/her hand and press ```key A``` (ensure Capslock to be OFF) when ready to collect data (the same ```key A``` is also used to ```pause and resume image image-capturing process```). 

**c)** The recording ```stops``` either user presses ```key Q``` (ensure Capslock to be OFF) on the keyboard or when the ```collecting.py``` script ```records 4000``` images of a particular label.

**d)** Once images corresponding to label W are collected, the user must go to the folder where those images are saved, and the user must select which images to keep and which ones to discard/delete. Once not-so-good images corresponding to label W are deleted, the user should re-run the above command of ```python collecting.py W```to capture more images to replace the deleted garbage images of label W. 

**g)** Once the user gathers ```4000 images``` each for all the required labels (like we did for one label of "W"), then he/she has to execute the following command:

```
python scripts/collecting.py final
```

This command will split those collected 4000 images of each label into 80:20 ratio for training and validation purposes and then save those images in a  predefined manner inside the ```final_dataset``` folder. This folder will be used during the training process.

### 1.3) Re-training on custom dataset

After collecting the custom dataset locally, run this script via Google Colab, [training.ipynb](https://github.com/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) and follow each instruction in this script. 

This script will clone this Github repo, ask you (the user) to put the custom dataset (located in the local system) on GDrive and will then train the underlying model. After training, the weights file gets saved in the ```TalkingHand``` folder (present in Google Drive), and then we can use it locally for inference (exactly like we did in section 1.2).
