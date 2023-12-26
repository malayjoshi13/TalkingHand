# TalkingHand
TalkingHand is a Computer Vision and Deep Learning-based **Sign Language to Text conversion system** which with the help of fine-tuned **convolutional neural network** of **VGG16**, classifies and converts the hand gestures made by the user into corresponding text-based labels. Custom dataset of about 4000 images each for 6 labels (i.e. A, B, C, D, SPACE, DELETE) has been collected for fine-tuning VGG-16 using a combination of ```background subtraction (createBackgroundSubtractorMOG2)``` and ```color threshold``` techniques. These techniques are used so that data collected will have a lower bias due to the shape & colour of the user's hand making the gesture and altering lighting conditions.

Got the following quantitative results after fine-tuning VGG16 on custom dataset: <br>
| | Train | Validation | Test |
| --- | --- | --- | --- |
| Loss | 0.0188 | 0.0913 |
| Accuracy | 0.9965 | 0.9888 | 0.802 |
| Precision | - | - | 0.805 |
| Recall | - | - | 0.801 |
| F1 Score | - | - | 0.803 |

The objective of this system is to help people with speaking and hearing disability to communicate with other people. Not choosing any specific sign language convention was intentional as with this experiment I wanted to understand the process to build a general hand sign recognizer which in future could be used for any conventional hand gestures.

https://github.com/malayjoshi13/TalkingHand/assets/71775151/e7c0a1b0-2afc-4f08-9015-8f764f3aa56b

## Setting up environment on the local system for **"inference"** and **"collecting custom dataset"**:

```
git clone https://github.com/malayjoshi13/TalkingHand.git

cd TalkingHand

conda create -n talkinghand_env

conda activate talkinghand_env

pip install -r requirements.txt
```

## Inference (Hand Gesture to Text)

Now download weights from [this link](https://drive.google.com/file/d/19tynPMUW8Ee9geskABT6QXKPkfXm9OiI/view?usp=sharing) and place it inside the "TalkingHand" folder present in your local system. After this, run the following command:

```
python scripts/gesture_to_text.py
```

**a)** On executing the above command, a window will pop up. 
 
**b)** This window is to only capture the background. So ensure your hand is not there within the bounding box and then press ```key B``` to take a picture of the background. 

**c)** Then another window will pop up, there you can make hand gestures to execute the following actions:

![instructions](https://github.com/malayjoshi13/TalkingHand/assets/71775151/3e1d6d1e-e552-488f-96ca-2672968e122b)

a) printing character "A" <br>
b) printing character "B" <br>
c) printing character "C" <br>
d) printing character "D" <br>
e) creating space between two adjacent character(s) <br>
f) deleting the character(s) <br>

## Collecting custom dataset for re-training
**a)** Training and validation data corresponding to 6 types of hand gestures is collected using the ```collecting.py``` script. For example, data for label ```W``` can be collected by executing the following command,

```
python scripts/collecting.py W
``` 

**b)** After executing the command, a window with a box on its right side will appear. First user has to press ```key B``` (ensure Capslock to be OFF) to capture background. During this time user should not plca hand within blue bounding box. 

**c)** After this user is supposed to keep his/her hand and press ```key A``` (ensure Capslock to be OFF) when ready to collect data (the same ```key A``` is also used to ```pause and resume image image-capturing process```). 

**d)** The recording ```stops``` either user presses ```key Q``` (ensure Capslock to be OFF) on the keyboard or when the ```collecting.py``` script ```records 4000``` images of a particular label.

**e)** Once images corresponding to label W are collected, the user must go to the folder where those images are saved, and the user must select which images to keep and which ones to discard/delete. Once not-so-good images corresponding to label W are deleted, the user should re-run the above command of ```python collecting.py W```to capture more images to replace the deleted garbage images of label W. 

**f)** Once the user gathers ```4000 images``` each for all the required labels (like we did for one label of "W"), then he/she has to execute the following command:

```
python scripts/collecting.py final
```

This command will split those collected 4000 images of each label into 80:20 ratio for training and validation purposes and then save those images in a  predefined manner inside the ```final_dataset``` folder. This folder will be used during the training process.

## Re-training - on your custom dataset, your set of hyperparameter values or both

After collecting the custom dataset locally, run this script via Google Colab, [training.ipynb](https://github.com/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malayjoshi13/TalkingHand/blob/main/scripts/training.ipynb) and follow each instruction in this script. 

This script will clone this Github repo, ask you (the user) to put the custom dataset (located in the local system) on GDrive and will then train the underlying model. After training, the weights file gets saved in the ```TalkingHand``` folder (present in Google Drive), and then we can use it locally for inference (exactly like we did in section 1.2).

In case you want to re-train on the dataset custom-created by me, just paste this link (https://drive.google.com/drive/folders/1NgiYx9NJ5h8ggXOp6s3VaT6G9wnGugEC?usp=drive_link) on your browser, create a shortcut of this dataset in your GDrive and use it with the above script.

## End-note
Thank you for patiently reading till here. I am pretty sure just like me, you would have also learned something new about object classification task on a real-world use case of building a system for detecting hand gestures and predicting their corresponding labels. Using these learned concepts, I will push myself to continue improving this tool. I encourage you also to do a comparative analysis of other CNN models and train on more diversed dataset to improve this tool!!

## Contributing
You are welcome to contribute to the repository with your PRs. In case of query or feedback, please write to me at 13.malayjoshi@gmail.com or https://www.linkedin.com/in/malayjoshi13/.

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/malayjoshi13/TalkingHand/blob/main/LICENSE)
