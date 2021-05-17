# TalkingHand
TalkingHand is a machine learning based conversation system which by help trained weights of a ***convolutional neural network*** called as ***VGG16 model*** converts the hand gestures made by user into corresponding text based classifying label of those hand gestures. 

Users can further extend this system on their own set of hand gestures. Will discuss the process below in upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gesture made by user and then classify it to the corresponding label. To do so this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it, so as to understand different features of training images and then to map and memorize those features with the provided label of that training image.



In this way the training process is carried out and the model layers of VGG16 + some additional layers are used together to memorize the mapping between features of images and their corresponding labels. Once done these memorizations are saved as in form of file called as ```weight file``` in ```hdf5``` format.

## 2) Usage:

before doing any step just do following things:

in git bash type ```cd /c/Users/barcode/Desktop```
then ```git clone https://github.com/malayjoshi13/TalkingHand.git```

then open your CMD and type: ```cd C:\Users\barcode\Desktop\TalkingHand```
then type: ```conda env create -f environment.yml```
then type: ```conda activate TalkingHand```







