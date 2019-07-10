# Dog-Breed-Classifier-Deep-Learning
Convolutional Neural Networks

# Project Overview:
This is Convolutional Neural Networks(CNN) project in the Deep Learning Nanodegree program of Udacity. In this project I learned how to develop algorithm that processe real-world images supplied by users and it's could be used as a web or mobile application. My algorithm identifies dog image that supplied by user into it's breed, and if the user supply a human image the algorithm identifies the resembling dog breed. 

For Example:
<img width="456" alt="Screen Shot 2019-07-10 at 11 33 36 PM" src="https://user-images.githubusercontent.com/46428156/61003051-a82f2180-a36b-11e9-8c4c-dc752e37c5c7.png">


# Project Instruction:
## Instructions
1. Clone the repository and navigate to the downloaded folder.

2. Open the Dog-breed Classifier_EN.ipynb(English Version)file. also you can find HTML version of the file.

3. Read and follow the instructions! This repository doesn't include any dataset you need. You can check out the notebook to download them.


# Project Information:
## Contents

- Step 0: Import Datasets 
- Step 1: Detect Humans 
- Step 2: Detect Dogs 
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch) 
- Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning) 
- Step 5: Write your Algorithm 
- Step 6: Test Your Algorithm

## Main CNN Model:
I had tried the pre-trained model (VGG-16) in step 3 and in step 4 for the transfer learning.

**VGG-16:**
is a convolutional Neural Network model that achieves 92.7% accuracy in ImageNet which roposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper (Very Deep Convolutional Networks for Large-Scale Image Recognition).
*Paper: https://paperswithcode.com/paper/very-deep-convolutional-networks-for-large

**The Architecture**
- Five convolutional blocks (2D conv.)
- 3 x 3 reseptive filed.
- ReLU as Activation function .
- Max Pooling.
- Classifier function:
- 3 FC layers at the top of the network.

![Uploading modified_vgg16.jpg…]()
Image Resource: https://www3.cs.stonybrook.edu/~zekzhang/cnn_classifier.html

## Dataset:
In this project used ImageNet. It is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories.

# Libraries 

The list below represents main libraries and its objects for the project.

- PyTorch (Convolutional Neural Network).
- OpenCV (Human Face Detection).
- Matplotlib (Plot Images).
- Numpy. 


# Accelerating the traing process:
In the training step in the Step 3 and 4, it is taking too long to run so you will need to either reduce the complexity of the VGG-16 architecture or switch to running the code on a GPU.

You can use **Amazon Web Services** to launch an EC2 GPU instance. (**But this costs money!**)

