# TalkingHand
TalkingHand is a Computer Vision and Deep Learning-based **Sign Language to Text conversion system** which with the help of fine-tuned **convolutional neural network** of **VGG16**, classifies and converts the hand gestures made by the user into corresponding text-based labels. Custom dataset of about 4000 images each for 6 labels (i.e. A, B, C, D, SPACE, DELETE) has been collected for fine-tuning the CNN model using a combination of ```background subtraction (createBackgroundSubtractorMOG2)``` and ```color threshold``` techniques so that data collected will have a lower bias due to the shape & colour of user's hand making the gesture and altering lighting conditions.

<br>

Got the following results after fine-tuning VGG16 on custom dataset: <br>
train loss: 0.0188, <br>
train accuracy: 0.9965, <br>
validation loss: 0.0913, <br> 
validation accuracy: 0.9888

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

Now download weights from [this link](https://drive.google.com/file/d/19tynPMUW8Ee9geskABT6QXKPkfXm9OiI/view?usp=sharing) and place it inside the "TalkingHand" folder present in your local system. After this, run the following command:

```
python scripts/gesture_to_text.py
```

**a)** On executing the above command, a window will pop up. 
 
![instructions](https://github.com/malayjoshi13/TalkingHand/assets/71775151/3e1d6d1e-e552-488f-96ca-2672968e122b)

**b)** This window is to capture background. So don't keep your hand in bounding box and press ```key B``` to take picture of background. 

**c)** Then another window will pop up, there you can make hand gestures to execute the following actions:

a) printing character "A" <br>
b) printing character "B" <br>
c) printing character "C" <br>
d) printing character "D" <br>
e) creating space between two adjacent character(s) <br>
f) deleting the character(s) <br>

### 1.3) Collecting custom dataset for re-training
**a)** Training and validation data corresponding to 6 types of hand gestures is collected using the ```collecting.py``` script. For example, data for label ```W``` can be collected by executing the following command,

```
python scripts/collecting.py W
``` 

**b)** After executing the command, a window with a box on its right side will appear. First user has to press ```key B``` (ensure Capslock to be OFF) to capture background. During this time user should not plca hand within blue bounding box. 

**c)** After this user is supposed to keep his/her hand and press ```key A``` (ensure Capslock to be OFF) when ready to collect data (the same ```key A``` is also used to ```pause and resume image image-capturing process```). 

**d)** The recording ```stops``` either user presses ```key Q``` (ensure Capslock to be OFF) on the keyboard or when the ```collecting.py``` script ```records 4000``` images of a particular label.

*e)** Once images corresponding to label W are collected, the user must go to the folder where those images are saved, and the user must select which images to keep and which ones to discard/delete. Once not-so-good images corresponding to label W are deleted, the user should re-run the above command of ```python collecting.py W```to capture more images to replace the deleted garbage images of label W. 

**f)** Once the user gathers ```4000 images``` each for all the required labels (like we did for one label of "W"), then he/she has to execute the following command:

```
python scripts/collecting.py final
```

This command will split those collected 4000 images of each label into 80:20 ratio for training and validation purposes and then save those images in a  predefined manner inside the ```final_dataset``` folder. This folder will be used during the training process.

### 1.3) Re-training - on your custom dataset, your own hyperparameter values or both

After collecting the custom dataset locally, run this script via Google Colab, [training.ipynb](https://github.com/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) and follow each instruction in this script. 

This script will clone this Github repo, ask you (the user) to put the custom dataset (located in the local system) on GDrive and will then train the underlying model. After training, the weights file gets saved in the ```TalkingHand``` folder (present in Google Drive), and then we can use it locally for inference (exactly like we did in section 1.2).

In case you want to re-train on the dataset custom-created by me, just paste this link (https://drive.google.com/drive/folders/1NgiYx9NJ5h8ggXOp6s3VaT6G9wnGugEC?usp=drive_link) on your browser, create a shortcut of this dataset in your GDrive and use it with the above script.
