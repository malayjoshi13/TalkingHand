# TalkingHand
TalkingHand is a machine learning based conversation system which by help trained weights of a ***convolutional neural network*** called as ***VGG16 model*** converts the hand gestures made by user into corresponding text based classifying label of those hand gestures. 

Users can further extend this system on their own set of hand gestures. Will discuss the process below in upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gesture made by user and then classify it to the corresponding label. To do so this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it, so as to understand different features of training images and then to map and memorize those features with the provided label of that training image.

![bandicam 2021-05-17 20-22-30-411](https://user-images.githubusercontent.com/71775151/118512165-30449600-b750-11eb-93e5-1a0724a8374c.jpg)

In this way the training process is carried out and the model layers of VGG16 + some additional layers are used together to memorize the mapping between features of images and their corresponding labels. Once done these memorizations are saved as in form of file called as ```weight file``` in ```hdf5``` format.

These weights are then used in future to classify a new and unseen image of hand gestures of users.
<br>
<br>

## 2) Usage

### 2.1) Prediction
**2.1.1)** Install Miniconda in your system from link:- https://docs.conda.io/en/latest/miniconda.html. (Tip: Keep agreeing and allowing blindly to what all prompts and buttons come in the process of installing miniconda with an exception case where you have to also tick the option to add miniconda to environment variable, i.e.:

Before

![Inked11aGz_LI (1)](https://user-images.githubusercontent.com/71775151/118517428-d1cde680-b754-11eb-88ec-edb6388063c3.jpg)

After

![install_python_path](https://user-images.githubusercontent.com/71775151/118516836-4f452700-b754-11eb-998e-6d96f56b9aed.png)

Also install Git Bash from link:- https://git-scm.com/downloads.

**2.1.2)** Open Git Bash and type following code in it to clone this GitHub repository at any location of your wish.

**Note:** Copy address of location where you want to clone this repository and paste it in format- "cd path_of_location"
```
cd /c/Users/thispc/Desktop
```
then,
```
git clone https://github.com/malayjoshi13/Describe.git
```

before doing any step just do following things:

in git bash type ```cd /c/Users/barcode/Desktop```
then ```git clone https://github.com/malayjoshi13/TalkingHand.git```

then open your CMD and type: ```cd C:\Users\barcode\Desktop\TalkingHand```
then type: ```conda env create -f environment.yml```
then type: ```conda activate TalkingHand```







