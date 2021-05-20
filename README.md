# TalkingHand
TalkingHand is a machine learning based conversation system which by help of trained weights of a **convolutional neural network* called as *VGG16 model** classifies and converts the hand gestures made by user into corresponding text based labels of those hand gestures. 

Users can further extend this system on their own set of hand gestures which will be discussed in the upcoming sections.

The objective of this system is to help people with speaking and hearing disability to communicate with other people.

## 1) What makes TalkingHand work?
TalkingHand uses Computer Vision technology (a branch of Deep Learning technology) to understand the hand gestures made by user and then classify them to the corresponding labels. To do so, this technology uses architecture of a type of Convolutional Neural Network known as VGG16 along with few more layers added over it. As a result it will understand different features of training images and then will map and memorize those features with their corresponding training labels.

![bandicam 2021-05-17 20-22-30-411](https://user-images.githubusercontent.com/71775151/118512165-30449600-b750-11eb-93e5-1a0724a8374c.jpg)

In this way the training process is carried out and the model layers of VGG16 + some additional layers are used together to memorize the mapping between features of images and their corresponding labels. Once done these memorizations are saved as in form of file called as weight file in hdf5 forma and then are used in future to classify a new and unseen scene consisting of hand gesture of user.
<br>
<br>

## 2) Usage

### 2.1) Setup for using the system
**2.1.1)** Install Miniconda in your system from link:- https://docs.conda.io/en/latest/miniconda.html. 

**Tip:** Keep agreeing and allowing blindly to what all prompts and buttons come in the process of installing miniconda with an exception case where you have to also tick the option to add miniconda to environment variable, i.e.:

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

Then execute,

```
git clone https://github.com/malayjoshi13/TalkingHand.git
```

**2.1.3)** Now type following code in your command prompt (CMD) to change you current working directory/location to location of "TalkingHand" folder (name of the cloned repository on your local system)

**Note:** Copy address of "TalkingHand" folder and paste it in format- "cd path_of_location"

```
cd C:\Users\thispc\Desktop\TalkingHand
```

Then execute,

```
conda env create -f environment.yml
```

above command creates virtual environment named "TalkingHand" (as present at top of "environment.yml") and also install packages mentioned in the yml file. After this execute,

```
conda activate TalkingHand
```
<br><br>
### 2.2) Prediction
Once setup is installed and virtual environment is created using the command prompt (CMD), execute following command in CMD,

```
python testing.py
```
On executing above command, a window will pop up where you can do hand gestures to execute following actions: <br>
a) printing character "A" <br>
b) printing character "B" <br>
c) printing character "C" <br>
d) deleting the characters <br>
e) creating space between two adjacent characters <br>
f) converting character(s) into audio <br>

Although current set of instructions are small in numbers but these can be further extended as per the need of user. The process to do so will be discussed in upcoming sections.
<br><br>
### 2.3) Data Collection
**2.3.1)** Using ```collection.py``` python file, training and validation data corresponding to 7 types of hand gestures namely is collected to train the model to get the ```weight``` which is used above for predicting the hand gestures of user. This collected data is then used to train the model using ```training.ipynb``` file (will see it in upcoming section).

One very ```interesting fact``` is that no matter initially on what training labels you have trained your model via ```training.py``` file, still during the prediction time via ```testing.py``` file you can change the name of labels according to your need.

In simple words, this can be explained like initially while training model using ```training.py``` my training labels were ```0```, ```1```, ```2```, ```3```, ```4```, ```5``` and ```nothing```. But later on while predicting labels for hand gestures, I changed the labels to ```A```, ```B```, ```C```, ```SPACE```, ```DELETE```, ```SPEAK``` and ```NOTHING```.

This was possible because model do not take labels in form of name based labels but take and remember labels with a sequenced indexing. Thus during prediction time we simply changed the mapping between ordered indexing and previous label names to same previous ordered indexing and new label names.

Thus in this way data is collected for getting the weights which we used above.

**2.3.2)** Now let us understand how can we follow the same process and collect data for more such labels.
a) Execute following command in CMD,

```




