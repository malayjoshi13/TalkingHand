# TalkingHand
TalkingHand is a machine learning based conversation system which by help of trained weights of a **convolutional neural network* called as *VGG16 model** classifies and converts the hand gestures made by user into corresponding text based labels of those hand gestures. 

Users can further extend this system on their own set of hand gestures which will be discussed in the upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gestures made by user and then classify them to the corresponding labels. To do so this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it. As a result it will understand different features of training images and then will map and memorize those features with their corresponding training labels.

![bandicam 2021-05-17 20-22-30-411](https://user-images.githubusercontent.com/71775151/118512165-30449600-b750-11eb-93e5-1a0724a8374c.jpg)

In this way the training process is carried out and the model layers of VGG16 + some additional layers are used together to memorize the mapping between features of images and their corresponding labels. Once done these memorizations are saved as in form of file called as weight file in hdf5 forma and then are used in future to classify a new and unseen scene consisting of hand gesture of user.
<br>
<br>

## 2) Usage

### 2.1) Setup for using the system
*2.1.1)* Install Miniconda in your system from link:- https://docs.conda.io/en/latest/miniconda.html. 

Tip: Keep agreeing and allowing blindly to what all prompts and buttons come in the process of installing miniconda with an exception case where you have to also tick the option to add miniconda to environment variable, i.e.:

Before

![Inked11aGz_LI (1)](https://user-images.githubusercontent.com/71775151/118517428-d1cde680-b754-11eb-88ec-edb6388063c3.jpg)

After

![install_python_path](https://user-images.githubusercontent.com/71775151/118516836-4f452700-b754-11eb-998e-6d96f56b9aed.png)

Also install Git Bash from link:- https://git-scm.com/downloads.

*2.1.2)* Open Git Bash and type following code in it to clone this GitHub repository at any location of your wish.

*Note:* Copy address of location where you want to clone this repository and paste it in format- "cd path_of_location"

```
cd /c/Users/thispc/Desktop
```

then,

```
git clone https://github.com/malayjoshi13/TalkingHand.git
```

**2.1.3)** Now type following code in your command prompt (CMD) to change you current working directory/location to location of "TalkingHand" folder (name of the cloned repository on your local system)

**Note:** Copy address of "TalkingHand" folder and paste it in format- "cd path_of_location"

```
cd C:\Users\thispc\Desktop\TalkingHand
```

then,

```
conda env create -f environment.yml
```

above command creates virtual environment named "TalkingHand" (as present at top of "environment.yml") and also install packages mentioned in the yml file, then execute,

```
conda activate TalkingHand
```

### 2.1) Prediction


then,

```
python testing.py
```
On executing above command, a window will pop up where you can do hand gestures to execute following actions:
a) printing character "A"
b) printing character "B"
c) printing character "C"
d) deleting the characters
e) creating space between two adjacent characters
f) converting character(s) into audio

Although current set of instructions are small in numbers but these can be further extended as per the need of user. The process to do so will be discussed in upcoming sections.

### 2.2) Data Collection




