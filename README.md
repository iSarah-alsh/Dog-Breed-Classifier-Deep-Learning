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

Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Write your Algorithm
Step 6: Test Your Algorithm

## Main CNN Model:
I had tried the pre-trained model (VGG-16) in step 3 and in step 4 for the transfer learning.

**VGG-16:**
is a convolutional Neural Network model that achieves 92.7% accuracy in ImageNet which roposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper (Very Deep Convolutional Networks for Large-Scale Image Recognition).
**The Architecture**
Five convolutional blocks (2D conv.)
3 x 3 reseptive filed.
ReLU as Activation function .
Max Pooling.
Classifier function:
3 FC layers at the top of the network.


## Dataset:
In this project used ImageNet. It is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories.

# Libraries 

The list below represents main libraries and its objects for the project.

PyTorch (Convolutional Neural Network)
OpenCV (Human Face Detection)
Matplotlib (Plot Images)
Numpy 


# Accelerating the traing process:
In the training step in the Step 3 and 4, it is taking too long to run so you will need to either reduce the complexity of the VGG-16 architecture or switch to running the code on a GPU.

You can use **Amazon Web Services** to launch an EC2 GPU instance. (**But this costs money!**)


Project: Write an Algorithm for a Dog Identification App
In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with '(IMPLEMENTATION)' in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
Note: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to File -> Download as -> HTML (.html). Include the finished document along with this notebook as your submission.
In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a 'Question X' header. Carefully read each question and provide thorough answers in the following text boxes that begin with 'Answer:'. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
Note: Code and Markdown cells can be executed using the Shift + Enter keyboard shortcut. Markdown cells can be edited by double-clicking the cell to enter edit mode.
The rubric contains optional "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.
Why We're Here
In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app. At the end of this project, your code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).
Sample Dog Output
In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed. There are many points of possible failure, and no perfect algorithm exists. Your imperfect solution will nonetheless create a fun user experience!
The Road Ahead
We break the notebook into separate steps. Feel free to use the links below to navigate the notebook.
Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Write your Algorithm
Step 6: Test Your Algorithm
Step 0: Import Datasets
Make sure that you've downloaded the required human and dog datasets:
Note: if you are using the Udacity workspace, you DO NOT need to re-download these - they can be found in the /data folder as noted in the cell below.
Download the dog dataset. Unzip the folder and place it in this project's home directory, at the location /dog_images.
Download the human dataset. Unzip the folder and place it in the home directory, at location /lfw.
Note: If you are using a Windows machine, you are encouraged to use 7zip to extract the folder.
In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays human_files and dog_files.
In [1]:
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
There are 13233 total human images.
There are 8351 total dog images.
Step 1: Detect Humans
In this section, we use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.
OpenCV provides many pre-trained face detectors, stored as XML files on github. We have downloaded one of these detectors and stored it in the haarcascades directory. In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.
In [2]:
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
Number of faces detected: 1

Before using any of the face detectors, it is standard procedure to convert the images to grayscale. The detectMultiScale function executes the classifier stored in face_cascade and takes the grayscale image as a parameter.
In the above code, faces is a numpy array of detected faces, where each row corresponds to a detected face. Each detected face is a 1D array with four entries that specifies the bounding box of the detected face. The first two entries in the array (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the bounding box. The last two entries in the array (extracted here as w and h) specify the width and height of the box.
Write a Human Face Detector
We can use this procedure to write a function that returns True if a human face is detected in an image and False otherwise. This function, aptly named face_detector, takes a string-valued file path to an image as input and appears in the code block below.
In [3]:
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
face_detector(human_files[0]) #will be return (True) if a human face is detected in an image and (False) otherwise
Out[3]:
True
(IMPLEMENTATION) Assess the Human Face Detector
Question 1: Use the code cell below to test the performance of the face_detector function.
What percentage of the first 100 images in human_files have a detected human face?
What percentage of the first 100 images in dog_files have a detected human face?
Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face. You will see that our algorithm falls short of this goal, but still gives acceptable performance. We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays human_files_short and dog_files_short.
Answer: The percentage of human faces pictures is = 99.0% , The percentage of human faces pictures detected in dog files is = 0.0%
In [4]:
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

for image in human_files_short:
    hum_pic = face_detector(image)
    sum_h = 100 - np.sum(hum_pic)
    hum_percentage = (sum_h / len(human_files_short)) * 100
print("The percentage of human faces pictures is = {}%".format(hum_percentage))


for image in dog_files_short:
    dog_pic = face_detector(image)
    dog_percentage = (np.sum(dog_pic) / len(dog_files_short)) * 100
print("The percentage of human faces pictures detected in dog files is = {}%".format(dog_percentage))
The percentage of human faces pictures is = 99.0%
The percentage of human faces pictures detected in dog files is = 0.0%
We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :). Please use the code cell below to design and test your own face detection algorithm. If you decide to pursue this optional task, report performance on human_files_short and dog_files_short.
In [8]:
### (Optional) 
### TODO: Test performance of anotherface detection algorithm.
### Feel free to use as many code cells as needed.
# Detect faces in the image
#------------------First Step: I chose to detect eyes using eyes detector ------------------
# draw a purple rectangle where the eye is detected
# extract pre-trained eyes detector
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find eyes in image
eye = eye_cascade.detectMultiScale(gray)

for (x,y,w,h) in eye:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(155,55,200),2)
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

#------------------Second Step: define our eye detector algorithm ------------------
def eye_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = eye_cascade.detectMultiScale(gray)
    return len(eye) > 0
eye_detector(human_files[0]) #will be return (True) if a human face is detected in an image and (False) otherwise

#------------------Third Step: Test the performance ------------------
for image in human_files_short:
    hum_pic = eye_detector(image)
    sum_h = 100 - np.sum(hum_pic)
    hum_percentage = (sum_h / len(human_files_short)) * 100
print("The percentage of human faces pictures is = {}%".format(hum_percentage))


for image in dog_files_short:
    dog_pic = eye_detector(image)
    dog_percentage = (np.sum(dog_pic) / len(dog_files_short)) * 100
print("The percentage of human faces pictures detected in dog files is = {}%".format(dog_percentage))

The percentage of human faces pictures is = 99.0%
The percentage of human faces pictures detected in dog files is = 1.0%
Step 2: Detect Dogs
In this section, we use a pre-trained model to detect dogs in images.
Obtain Pre-trained VGG-16 Model
The code cell below downloads the VGG-16 model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories.
In [5]:
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
100%|██████████| 553433881/553433881 [00:09<00:00, 60524307.46it/s]
Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.
(IMPLEMENTATION) Making Predictions with a Pre-trained Model
In the next code cell, you will write a function that accepts a path to an image (such as 'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg') as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model. The output should always be an integer between 0 and 999, inclusive.
Before writing the function, make sure that you take the time to learn how to appropriately pre-process tensors for pre-trained models in the PyTorch documentation.
In [6]:
from PIL import Image
import torchvision.transforms as transforms

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    predicted_class = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(p=0.02),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    To_tensor= predicted_class(Image.open(img_path)).unsqueeze(0).float()
    
    if use_cuda:
        To_tensor=To_tensor.cuda()
        
    pred_class=VGG16(To_tensor).data.argmax()

    return pred_class
print(VGG16_predict(dog_files[400]))
    
img = plt.imread(dog_files[400])
fig, ax = plt.subplots()
ax.imshow(img)
tensor(156, device='cuda:0')
Out[6]:
<matplotlib.image.AxesImage at 0x7f0cc4655cc0>

(IMPLEMENTATION) Write a Dog Detector
While looking at the dictionary, you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from 'Chihuahua' to 'Mexican hairless'. Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).
Use these ideas to complete the dog_detector function below, which returns True if a dog is detected in an image (and False if not).
In [10]:
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    pred_class=VGG16_predict(img_path)
    if(pred_class>=151 and pred_class<=268):
        return True
    else:
        return False
print(dog_detector(dog_files[350]))
True
(IMPLEMENTATION) Assess the Dog Detector
Question 2: Use the code cell below to test the performance of your dog_detector function.
What percentage of the images in human_files_short have a detected dog?
What percentage of the images in dog_files_short have a detected dog?
Answer: The percentage of dog pictures detected in human file is = 0.0% , The percentage of dog pictures detected in dog files is = 99.0%
In [9]:
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

for image in human_files_short:
    hum_pic = dog_detector(image)
    hum_percentage = (np.sum(hum_pic) / len(human_files_short)) * 100
print("The percentage of dog pictures detected in human file is = {}%".format(hum_percentage))

for image in dog_files_short:
    dog_pic = dog_detector(image)
    sum_d = 100 - np.sum(dog_pic)
    dog_percentage = (sum_d / len(dog_files_short)) * 100
print("The percentage of dog pictures detected in dog files is = {}%".format(dog_percentage))
The percentage of dog pictures detected in human file is = 0.0%
The percentage of dog pictures detected in dog files is = 99.0%
We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as Inception-v3, ResNet-50, etc). Please use the code cell below to test other pre-trained PyTorch models. If you decide to pursue this optional task, report performance on human_files_short and dog_files_short.
In [ ]:
### (Optional) 
### TODO: Report the performance of another pre-trained network.
### Feel free to use as many code cells as needed.
# define ResNet-50 model
ResNet50 = models.ResNet50(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    ResNet50 = ResNet50.cuda()
    
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images. In this step, you will create a CNN that classifies dog breeds. You must create your CNN from scratch (so, you can't use transfer learning yet!), and you must attain a test accuracy of at least 10%. In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.
We mention that the task of assigning breed to dogs from images is considered exceptionally challenging. To see why, consider that even a human would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.
Brittany	Welsh Springer Spaniel
	
It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).
Curly-Coated Retriever	American Water Spaniel
	
Likewise, recall that labradors come in yellow, chocolate, and black. Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.
Yellow Labrador	Chocolate Labrador	Black Labrador
		
We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.
Remember that the practice is far ahead of the theory in deep learning. Experiment with many different architectures, and trust your intuition. And, of course, have fun!
(IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
Use the code cell below to write three separate data loaders for the training, validation, and test datasets of dog images (located at dog_images/train, dog_images/valid, and dog_images/test, respectively). You may find this documentation on custom datasets to be a useful resource. If you are interested in augmenting your training and/or validation data, check out the wide variety of transforms!
In [9]:
import os
from torchvision import datasets
import torch
import torchvision.transforms as transforms


### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to a normalized torch.FloatTensor
transform = {'train':transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(p=0.02),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]),
             'valid':transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]),
             'test':transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
             }

# choose the training and test datasets
train_data = datasets.ImageFolder('/data/dog_images/train', transform=transform['train'])
test_data = datasets.ImageFolder('/data/dog_images/test', transform=transform['valid'])
valid_data = datasets.ImageFolder('/data/dog_images/valid', transform=transform['test'])

# prepare data loaders 
loaders_scratch = {'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers
                                                        , shuffle=True),
'valid': torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=False),
'test': torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers, shuffle=False)
}
Question 3: Describe your chosen procedure for preprocessing the data.
How does your code resize the images (by cropping, stretching, etc)? What size did you pick for the input tensor, and why?
Did you decide to augment the dataset? If so, how (through translations, flips, rotations, etc)? If not, why not?
Answer: I chose to resize my dataset by RandomResizedCrop with 224 for input's size also i did some augmentation to the train dataset like:randomly flip ,because i think it’s will help model to learn better and help to genaralizing also to obtain a good performance.
(IMPLEMENTATION) Model Architecture
Create a CNN to classify dog breed. Use the template in the code cell below.
In [10]:
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        # convolutional layers 
        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layers 
        self.fc1 = nn.Linear(128 * 28 * 28, 700) 
        self.fc2 = nn.Linear(700, 133)
        # dropout layer (p=0.6)
        self.dropout = nn.Dropout(0.6)
    
    def forward(self, x):
        ## Define forward behavior
        #Sequence of convolutional layers and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input (this step is important after sequence of CL & MaxPL to know how numbers of parameters )
        x = x.view(-1, 128 * 28 * 28)
        # add dropout layer (dropout layer to avoid overfitting)
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)

        return x

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
Question 4: Outline the steps you took to get to your final CNN architecture and your reasoning at each step.
Answer: My CNN architecture include: 3 convolutional layers (with stride=1 by defult and padding=1) with 3 Max pooling layers and 2 fully-connected layers and one dropout layer to avoid any overfitting may occur (with 60%). The behavior of my model will be as: input with size= 224x224x3 (depth=3,becuse it's RGB images)enters to the first convolutional layer (with 32 filters) the images will be output image with depth=32 then will be enter as input to the maxpooling layer (to reduce the high and wigth of image) after applying ReLU activation function (the purpose of the activation function is to scale the output of layer to make sure that model is trining more efficient), then after the thired convolutional layer the output will be flatten to enters to the fully-connected layers which have 700 nodes and will be use dropout layer ,finally the predicted output will be on of the true class (we have 133 classes) . I used stochastic gradient descent as optimizer with learning rate = 0.03 and categorical cross-entropy as loss function.
(IMPLEMENTATION) Specify Loss Function and Optimizer
Use the next code cell to specify a loss function and optimizer. Save the chosen loss function as criterion_scratch, and the optimizer as optimizer_scratch below.
In [11]:
import torch.optim as optim

### TODO: select loss function 
#I used categorical cross-entropy as loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
#I used stochastic gradient descent as optimizer with learning rate = 0.03
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.03)
In [12]:
#I Wrote this code due to i got an error when i run (train and validate code), and it's solve the problem
# Error message (image file is truncated (150 bytes not processed)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
(IMPLEMENTATION) Train and Validate the Model
Train and validate your model in the code cell below. Save the final model parameters at filepath 'model_scratch.pt'.
In [23]:
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    # initialize tracker for minimum validation loss
        """returns trained model"""
        valid_loss_min = np.Inf 
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
        ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
                optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
            # calculate the batch loss
                loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
            # perform a single optimization step (parameter update)
                optimizer.step()
            # update training loss
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
                if batch_idx%50==0:
                    print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch,train_loss))
        ######################    
        # validate the model #
        ######################
            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
               # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
            
            
            
            # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
            # calculate the batch loss
                loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            #loss.backward()
            # perform a single optimization step (parameter update)
            #optimizer.step()
            # update average validation loss 
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'model_scratch.pt')
                valid_loss_min = valid_loss
                # return trained model
        return model
            
      # train the model
model_scratch = train(15, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')
    # load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
model_scratch
Epoch: 1 	Training Loss: 3.189580 
Epoch: 1 	Training Loss: 3.541442 
Epoch: 1 	Training Loss: 3.523133 
Epoch: 1 	Training Loss: 3.526014 
Epoch: 1 	Training Loss: 3.509018 
Epoch: 1 	Training Loss: 3.495072 
Epoch: 1 	Training Loss: 3.505710 
Epoch: 1 	Training Loss: 3.501925 	Validation Loss: 4.007047
Validation loss decreased (inf --> 4.007047).  Saving model ...
Epoch: 2 	Training Loss: 3.093534 
Epoch: 2 	Training Loss: 3.344998 
Epoch: 2 	Training Loss: 3.324170 
Epoch: 2 	Training Loss: 3.363513 
Epoch: 2 	Training Loss: 3.392816 
Epoch: 2 	Training Loss: 3.401857 
Epoch: 2 	Training Loss: 3.406613 
Epoch: 2 	Training Loss: 3.406494 	Validation Loss: 3.913435
Validation loss decreased (4.007047 --> 3.913435).  Saving model ...
Epoch: 3 	Training Loss: 3.262853 
Epoch: 3 	Training Loss: 3.337283 
Epoch: 3 	Training Loss: 3.323112 
Epoch: 3 	Training Loss: 3.308711 
Epoch: 3 	Training Loss: 3.316318 
Epoch: 3 	Training Loss: 3.331667 
Epoch: 3 	Training Loss: 3.338844 
Epoch: 3 	Training Loss: 3.344233 	Validation Loss: 4.008664
Epoch: 4 	Training Loss: 2.876757 
Epoch: 4 	Training Loss: 3.289010 
Epoch: 4 	Training Loss: 3.310202 
Epoch: 4 	Training Loss: 3.277625 
Epoch: 4 	Training Loss: 3.291775 
Epoch: 4 	Training Loss: 3.297774 
Epoch: 4 	Training Loss: 3.308776 
Epoch: 4 	Training Loss: 3.313314 	Validation Loss: 3.964009
Epoch: 5 	Training Loss: 2.992740 
Epoch: 5 	Training Loss: 3.116511 
Epoch: 5 	Training Loss: 3.198286 
Epoch: 5 	Training Loss: 3.204112 
Epoch: 5 	Training Loss: 3.196615 
Epoch: 5 	Training Loss: 3.189986 
Epoch: 5 	Training Loss: 3.212881 
Epoch: 5 	Training Loss: 3.212112 	Validation Loss: 4.026076
Epoch: 6 	Training Loss: 3.087085 
Epoch: 6 	Training Loss: 3.122618 
Epoch: 6 	Training Loss: 3.097343 
Epoch: 6 	Training Loss: 3.119027 
Epoch: 6 	Training Loss: 3.142292 
Epoch: 6 	Training Loss: 3.153025 
Epoch: 6 	Training Loss: 3.162305 
Epoch: 6 	Training Loss: 3.159275 	Validation Loss: 3.926233
Epoch: 7 	Training Loss: 2.570190 
Epoch: 7 	Training Loss: 3.121739 
Epoch: 7 	Training Loss: 3.124053 
Epoch: 7 	Training Loss: 3.093238 
Epoch: 7 	Training Loss: 3.097998 
Epoch: 7 	Training Loss: 3.113738 
Epoch: 7 	Training Loss: 3.126683 
Epoch: 7 	Training Loss: 3.143632 	Validation Loss: 4.090447
Epoch: 8 	Training Loss: 3.424042 
Epoch: 8 	Training Loss: 3.024025 
Epoch: 8 	Training Loss: 3.040359 
Epoch: 8 	Training Loss: 3.068742 
Epoch: 8 	Training Loss: 3.077374 
Epoch: 8 	Training Loss: 3.098987 
Epoch: 8 	Training Loss: 3.095263 
Epoch: 8 	Training Loss: 3.083702 	Validation Loss: 4.196418
Epoch: 9 	Training Loss: 2.861737 
Epoch: 9 	Training Loss: 2.968211 
Epoch: 9 	Training Loss: 2.958138 
Epoch: 9 	Training Loss: 2.985599 
Epoch: 9 	Training Loss: 3.014813 
Epoch: 9 	Training Loss: 3.010370 
Epoch: 9 	Training Loss: 3.014595 
Epoch: 9 	Training Loss: 3.011473 	Validation Loss: 4.241480
Epoch: 10 	Training Loss: 2.816602 
Epoch: 10 	Training Loss: 2.853145 
Epoch: 10 	Training Loss: 2.893171 
Epoch: 10 	Training Loss: 2.907961 
Epoch: 10 	Training Loss: 2.950670 
Epoch: 10 	Training Loss: 2.954258 
Epoch: 10 	Training Loss: 2.963510 
Epoch: 10 	Training Loss: 2.973976 	Validation Loss: 4.126119
Epoch: 11 	Training Loss: 2.562054 
Epoch: 11 	Training Loss: 2.782923 
Epoch: 11 	Training Loss: 2.804242 
Epoch: 11 	Training Loss: 2.835082 
Epoch: 11 	Training Loss: 2.856814 
Epoch: 11 	Training Loss: 2.885107 
Epoch: 11 	Training Loss: 2.894878 
Epoch: 11 	Training Loss: 2.914799 	Validation Loss: 4.029228
Epoch: 12 	Training Loss: 2.720857 
Epoch: 12 	Training Loss: 2.826984 
Epoch: 12 	Training Loss: 2.840555 
Epoch: 12 	Training Loss: 2.832189 
Epoch: 12 	Training Loss: 2.833119 
Epoch: 12 	Training Loss: 2.850529 
Epoch: 12 	Training Loss: 2.843113 
Epoch: 12 	Training Loss: 2.848487 	Validation Loss: 4.038024
Epoch: 13 	Training Loss: 2.566741 
Epoch: 13 	Training Loss: 2.726265 
Epoch: 13 	Training Loss: 2.781346 
Epoch: 13 	Training Loss: 2.756881 
Epoch: 13 	Training Loss: 2.775945 
Epoch: 13 	Training Loss: 2.786482 
Epoch: 13 	Training Loss: 2.794691 
Epoch: 13 	Training Loss: 2.802514 	Validation Loss: 4.127317
Epoch: 14 	Training Loss: 3.146820 
Epoch: 14 	Training Loss: 2.718348 
Epoch: 14 	Training Loss: 2.708208 
Epoch: 14 	Training Loss: 2.726138 
Epoch: 14 	Training Loss: 2.731055 
Epoch: 14 	Training Loss: 2.757859 
Epoch: 14 	Training Loss: 2.770093 
Epoch: 14 	Training Loss: 2.781174 	Validation Loss: 4.009572
Epoch: 15 	Training Loss: 2.759409 
Epoch: 15 	Training Loss: 2.646099 
Epoch: 15 	Training Loss: 2.673147 
Epoch: 15 	Training Loss: 2.699324 
Epoch: 15 	Training Loss: 2.717550 
Epoch: 15 	Training Loss: 2.723384 
Epoch: 15 	Training Loss: 2.722303 
Epoch: 15 	Training Loss: 2.722401 	Validation Loss: 4.116002
Out[23]:
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=100352, out_features=700, bias=True)
  (fc2): Linear(in_features=700, out_features=133, bias=True)
  (dropout): Dropout(p=0.6)
)
(IMPLEMENTATION) Test the Model
Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy. Ensure that your test accuracy is greater than 10%.
In [24]:
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders_scratch['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
Test Loss: 4.074601


Test Accuracy: 11% (92/836)
Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
You will now use transfer learning to create a CNN that can identify dog breed from images. Your CNN must attain at least 60% accuracy on the test set.
(IMPLEMENTATION) Specify Data Loaders for the Dog Dataset
Use the code cell below to write three separate data loaders for the training, validation, and test datasets of dog images (located at dogImages/train, dogImages/valid, and dogImages/test, respectively).
If you like, you are welcome to use the same data loaders from the previous step, when you created a CNN from scratch.
In [13]:
## TODO: Specify data loaders
#I used the same data loaders from the previous step
data_loders_t = loaders_scratch.copy()
(IMPLEMENTATION) Model Architecture
Use transfer learning to create a CNN to classify dog breed. Use the code cell below, and save your initialized model as the variable model_transfer.
In [14]:
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 
# define Resnet50 model
model_transfer = models.resnet50(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = True
    
model_transfer.fc = nn.Linear(2048, 133) #fully connected layer with 2048 features and 133 classes
full_conn_parameters = model_transfer.fc.parameters()
for param in full_conn_parameters:
    param.requires_grad = True

    # check if CUDA is available
use_cuda = torch.cuda.is_available()

print(model_transfer)
if use_cuda:
    model_transfer = model_transfer.cuda()
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /root/.torch/models/resnet50-19c8e357.pth
100%|██████████| 102502400/102502400 [00:06<00:00, 15789074.15it/s]
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=2048, out_features=133, bias=True)
)
Question 5: Outline the steps you took to get to your final CNN architecture and your reasoning at each step. Describe why you think the architecture is suitable for the current problem.
Answer: I used Resent 50 after i tried VGG16 and it's gave me a good accuracy (68%) it's a good pre-trined for this dataset, the final CNN architecture is i added a fully connected layer with 2048 features and 133 classes.
(IMPLEMENTATION) Specify Loss Function and Optimizer
Use the next code cell to specify a loss function and optimizer. Save the chosen loss function as criterion_transfer, and the optimizer as optimizer_transfer below.
In [15]:
import torch.optim as optim
#I used categorical cross-entropy as loss function
criterion_transfer = nn.CrossEntropyLoss()

### TODO: select optimizer
#I used stochastic gradient descent as optimizer with learning rate = 0.03 
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.03)
(IMPLEMENTATION) Train and Validate the Model
Train and validate your model in the code cell below. Save the final model parameters at filepath 'model_transfer.pt'.
In [16]:
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
In [17]:
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    # initialize tracker for minimum validation loss
        """returns trained model"""
        valid_loss_min = np.Inf 
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
        ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            # clear the gradients of all optimized variables
                optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
            # calculate the batch loss
                loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
            # perform a single optimization step (parameter update)
                optimizer.step()
            # update training loss
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
                if batch_idx%50==0:
                    print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch,train_loss))
        ######################    
        # validate the model #
        ######################
            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
               # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
            
            
            
            # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
            # calculate the batch loss
                loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            #loss.backward()
            # perform a single optimization step (parameter update)
            #optimizer.step()
            # update average validation loss 
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'model_transfer.pt')
                valid_loss_min = valid_loss
                # return trained model
        return model
            
      # train the model
model_transfer = train(20, data_loders_t, model_transfer, optimizer_transfer, 
                      criterion_transfer, use_cuda, 'model_transfer.pt')
    # load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
model_transfer
Epoch: 1 	Training Loss: 4.860424 
Epoch: 1 	Training Loss: 4.390319 
Epoch: 1 	Training Loss: 3.716216 
Epoch: 1 	Training Loss: 3.248189 
Epoch: 1 	Training Loss: 2.908597 
Epoch: 1 	Training Loss: 2.664391 
Epoch: 1 	Training Loss: 2.488953 
Epoch: 1 	Training Loss: 2.391906 	Validation Loss: 1.231088
Validation loss decreased (inf --> 1.231088).  Saving model ...
Epoch: 2 	Training Loss: 1.945613 
Epoch: 2 	Training Loss: 1.344330 
Epoch: 2 	Training Loss: 1.316264 
Epoch: 2 	Training Loss: 1.313259 
Epoch: 2 	Training Loss: 1.310799 
Epoch: 2 	Training Loss: 1.287207 
Epoch: 2 	Training Loss: 1.282939 
Epoch: 2 	Training Loss: 1.276144 	Validation Loss: 1.581760
Epoch: 3 	Training Loss: 1.249697 
Epoch: 3 	Training Loss: 1.109908 
Epoch: 3 	Training Loss: 1.110413 
Epoch: 3 	Training Loss: 1.099555 
Epoch: 3 	Training Loss: 1.083316 
Epoch: 3 	Training Loss: 1.076159 
Epoch: 3 	Training Loss: 1.077386 
Epoch: 3 	Training Loss: 1.084209 	Validation Loss: 1.349078
Epoch: 4 	Training Loss: 1.276784 
Epoch: 4 	Training Loss: 1.009606 
Epoch: 4 	Training Loss: 0.968753 
Epoch: 4 	Training Loss: 0.995696 
Epoch: 4 	Training Loss: 0.972367 
Epoch: 4 	Training Loss: 0.976847 
Epoch: 4 	Training Loss: 0.980813 
Epoch: 4 	Training Loss: 0.982464 	Validation Loss: 1.265775
Epoch: 5 	Training Loss: 0.759960 
Epoch: 5 	Training Loss: 0.879978 
Epoch: 5 	Training Loss: 0.906627 
Epoch: 5 	Training Loss: 0.890609 
Epoch: 5 	Training Loss: 0.898457 
Epoch: 5 	Training Loss: 0.913537 
Epoch: 5 	Training Loss: 0.907995 
Epoch: 5 	Training Loss: 0.917749 	Validation Loss: 1.032825
Validation loss decreased (1.231088 --> 1.032825).  Saving model ...
Epoch: 6 	Training Loss: 0.890768 
Epoch: 6 	Training Loss: 0.860912 
Epoch: 6 	Training Loss: 0.807157 
Epoch: 6 	Training Loss: 0.838771 
Epoch: 6 	Training Loss: 0.832845 
Epoch: 6 	Training Loss: 0.852182 
Epoch: 6 	Training Loss: 0.852496 
Epoch: 6 	Training Loss: 0.851574 	Validation Loss: 0.966277
Validation loss decreased (1.032825 --> 0.966277).  Saving model ...
Epoch: 7 	Training Loss: 1.016260 
Epoch: 7 	Training Loss: 0.813344 
Epoch: 7 	Training Loss: 0.795057 
Epoch: 7 	Training Loss: 0.796408 
Epoch: 7 	Training Loss: 0.811295 
Epoch: 7 	Training Loss: 0.804128 
Epoch: 7 	Training Loss: 0.799876 
Epoch: 7 	Training Loss: 0.798339 	Validation Loss: 1.295624
Epoch: 8 	Training Loss: 1.110998 
Epoch: 8 	Training Loss: 0.764187 
Epoch: 8 	Training Loss: 0.777593 
Epoch: 8 	Training Loss: 0.762565 
Epoch: 8 	Training Loss: 0.753376 
Epoch: 8 	Training Loss: 0.758880 
Epoch: 8 	Training Loss: 0.750747 
Epoch: 8 	Training Loss: 0.746483 	Validation Loss: 1.069316
Epoch: 9 	Training Loss: 0.375882 
Epoch: 9 	Training Loss: 0.720497 
Epoch: 9 	Training Loss: 0.699687 
Epoch: 9 	Training Loss: 0.707226 
Epoch: 9 	Training Loss: 0.696200 
Epoch: 9 	Training Loss: 0.719210 
Epoch: 9 	Training Loss: 0.728915 
Epoch: 9 	Training Loss: 0.734166 	Validation Loss: 1.061957
Epoch: 10 	Training Loss: 0.957414 
Epoch: 10 	Training Loss: 0.703703 
Epoch: 10 	Training Loss: 0.690795 
Epoch: 10 	Training Loss: 0.695486 
Epoch: 10 	Training Loss: 0.676042 
Epoch: 10 	Training Loss: 0.675997 
Epoch: 10 	Training Loss: 0.680345 
Epoch: 10 	Training Loss: 0.683551 	Validation Loss: 1.066582
Epoch: 11 	Training Loss: 0.366662 
Epoch: 11 	Training Loss: 0.627488 
Epoch: 11 	Training Loss: 0.638210 
Epoch: 11 	Training Loss: 0.669997 
Epoch: 11 	Training Loss: 0.667201 
Epoch: 11 	Training Loss: 0.670962 
Epoch: 11 	Training Loss: 0.672676 
Epoch: 11 	Training Loss: 0.677584 	Validation Loss: 1.142742
Epoch: 12 	Training Loss: 0.697251 
Epoch: 12 	Training Loss: 0.686364 
Epoch: 12 	Training Loss: 0.648630 
Epoch: 12 	Training Loss: 0.632926 
Epoch: 12 	Training Loss: 0.640606 
Epoch: 12 	Training Loss: 0.632381 
Epoch: 12 	Training Loss: 0.633077 
Epoch: 12 	Training Loss: 0.634886 	Validation Loss: 1.205375
Epoch: 13 	Training Loss: 0.364329 
Epoch: 13 	Training Loss: 0.530274 
Epoch: 13 	Training Loss: 0.558523 
Epoch: 13 	Training Loss: 0.606140 
Epoch: 13 	Training Loss: 0.620243 
Epoch: 13 	Training Loss: 0.619044 
Epoch: 13 	Training Loss: 0.616303 
Epoch: 13 	Training Loss: 0.617364 	Validation Loss: 1.051126
Epoch: 14 	Training Loss: 0.432446 
Epoch: 14 	Training Loss: 0.556402 
Epoch: 14 	Training Loss: 0.551198 
Epoch: 14 	Training Loss: 0.574974 
Epoch: 14 	Training Loss: 0.587420 
Epoch: 14 	Training Loss: 0.588183 
Epoch: 14 	Training Loss: 0.589215 
Epoch: 14 	Training Loss: 0.589657 	Validation Loss: 1.128373
Epoch: 15 	Training Loss: 0.688318 
Epoch: 15 	Training Loss: 0.558828 
Epoch: 15 	Training Loss: 0.553972 
Epoch: 15 	Training Loss: 0.587723 
Epoch: 15 	Training Loss: 0.580400 
Epoch: 15 	Training Loss: 0.585630 
Epoch: 15 	Training Loss: 0.580484 
Epoch: 15 	Training Loss: 0.578342 	Validation Loss: 1.068228
Epoch: 16 	Training Loss: 0.352696 
Epoch: 16 	Training Loss: 0.469320 
Epoch: 16 	Training Loss: 0.518720 
Epoch: 16 	Training Loss: 0.511631 
Epoch: 16 	Training Loss: 0.531837 
Epoch: 16 	Training Loss: 0.545429 
Epoch: 16 	Training Loss: 0.539092 
Epoch: 16 	Training Loss: 0.535122 	Validation Loss: 1.025915
Epoch: 17 	Training Loss: 0.134209 
Epoch: 17 	Training Loss: 0.533628 
Epoch: 17 	Training Loss: 0.500902 
Epoch: 17 	Training Loss: 0.512389 
Epoch: 17 	Training Loss: 0.521062 
Epoch: 17 	Training Loss: 0.536668 
Epoch: 17 	Training Loss: 0.545695 
Epoch: 17 	Training Loss: 0.553891 	Validation Loss: 1.086905
Epoch: 18 	Training Loss: 1.084270 
Epoch: 18 	Training Loss: 0.538407 
Epoch: 18 	Training Loss: 0.527954 
Epoch: 18 	Training Loss: 0.517093 
Epoch: 18 	Training Loss: 0.525342 
Epoch: 18 	Training Loss: 0.522369 
Epoch: 18 	Training Loss: 0.536047 
Epoch: 18 	Training Loss: 0.537112 	Validation Loss: 1.173403
Epoch: 19 	Training Loss: 0.417507 
Epoch: 19 	Training Loss: 0.547072 
Epoch: 19 	Training Loss: 0.538706 
Epoch: 19 	Training Loss: 0.529567 
Epoch: 19 	Training Loss: 0.527593 
Epoch: 19 	Training Loss: 0.530687 
Epoch: 19 	Training Loss: 0.529679 
Epoch: 19 	Training Loss: 0.527222 	Validation Loss: 1.061980
Epoch: 20 	Training Loss: 0.561132 
Epoch: 20 	Training Loss: 0.459838 
Epoch: 20 	Training Loss: 0.450357 
Epoch: 20 	Training Loss: 0.465704 
Epoch: 20 	Training Loss: 0.467909 
Epoch: 20 	Training Loss: 0.487836 
Epoch: 20 	Training Loss: 0.495490 
Epoch: 20 	Training Loss: 0.498276 	Validation Loss: 1.202254
Out[17]:
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=2048, out_features=133, bias=True)
)
(IMPLEMENTATION) Test the Model
Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy. Ensure that your test accuracy is greater than 60%.
In [18]:
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    # load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
# call test function   

test(data_loders_t, model_transfer, criterion_transfer, use_cuda)
Test Loss: 1.150282


Test Accuracy: 68% (572/836)
(IMPLEMENTATION) Predict Dog Breed with the Model
Write a function that takes an image path as input and returns the dog breed (Affenpinscher, Afghan hound, etc) that is predicted by your model.
In [19]:
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in data_loders_t['train'].dataset.classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    To_tensor= transform(Image.open(img_path)).unsqueeze(0).float()
    pred_class = False
    # check if use CUDA 
    if use_cuda:
        pred_class = model_transfer.forward(Variable(To_tensor.cuda())).cpu()
    else:
        pred_class = model_transfer.forward(Variable(To_tensor))
        
    return class_names[pred_class.data.numpy().argmax()]
In [20]:
predict_breed_transfer(dog_files[100])
Out[20]:
'German pinscher'
In [21]:
predict_breed_transfer(dog_files[300])
Out[21]:
'Chinese crested'
Step 5: Write your Algorithm
Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. Then,
if a dog is detected in the image, return the predicted breed.
if a human is detected in the image, return the resembling dog breed.
if neither is detected in the image, provide output that indicates an error.
You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the face_detector and human_detector functions developed above. You are required to use your CNN from Step 4 to predict dog breed.
Some sample output for our algorithm is provided below, but feel free to design your own user experience!
Sample Human Output
(IMPLEMENTATION) Write your Algorithm
In [22]:
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    dog_face = dog_detector(img_path)
    humen_face = face_detector(img_path)
    
    image = Image.open(img_path)
    plt.imshow(image)
    plt.show()
    
    if(humen_face):
        print('This is a humen face and it is look like a =' + predict_breed_transfer(img_path))
    elif (dog_face):
        print('This is a dog face and its breed is =' + predict_breed_transfer(img_path))
    else:
        print('There is no human or dog face in this picture')

run_app(dog_files[500])

This is a humen face and it is look like a =Papillon
Step 6: Test Your Algorithm
In this section, you will take your new algorithm for a spin! What kind of dog does the algorithm think that you look like? If you have a dog, does it predict your dog's breed accurately? If you have a cat, does it mistakenly think that your cat is a dog?
(IMPLEMENTATION) Test Your Algorithm on Sample Images!
Test your algorithm at least six images on your computer. Feel free to use any images you like. Use at least two human and two dog images.
Question 6: Is the output better than you expected :) ? Or worse :( ? Provide at least three possible points of improvement for your algorithm.
Answer: yes it is better than I expected, but it predicted wrong in one picture ('no-hum-no-dog.jpg') it prdectied as dog which is a leopard،So, I think my algorithm need some work to improve like Increase the number of epoch also I think some change in the CNN layres parameters may affect the result better.
In [23]:
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)

This is a humen face and it is look like a =Bull terrier

This is a humen face and it is look like a =Alaskan malamute

This is a humen face and it is look like a =Alaskan malamute

This is a dog face and its breed is =Greyhound

This is a dog face and its breed is =Mastiff

This is a dog face and its breed is =Bullmastiff
In [24]:
run_app('images/human1.jpg')

This is a humen face and it is look like a =Akita
In [25]:
run_app('images/human2.jpg')

This is a humen face and it is look like a =Chesapeake bay retriever
In [31]:
run_app('images/human3.jpeg')

This is a humen face and it is look like a =Poodle
In [27]:
run_app('images/dog1.jpg')

This is a dog face and its breed is =Australian cattle dog
In [28]:
run_app('images/dog2.jpg')

This is a dog face and its breed is =Icelandic sheepdog
In [29]:
run_app('images/dog3.jpg')

This is a dog face and its breed is =Alaskan malamute
In [30]:
run_app('images/dog4.jpeg')

This is a dog face and its breed is =Chesapeake bay retriever
In [35]:
run_app('images/no-hum-no-dog.jpg')

This is a humen face and it is look like a =Dalmatian
In [36]:
run_app('images/my-draw.jpg')

This is a humen face and it is look like a =Xoloitzcuintli
In [37]:
run_app('images/japan.jpg')

There is no human or dog face in this picture
