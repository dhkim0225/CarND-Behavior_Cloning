**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/left_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image2]: ./images/center_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image3]: ./images/right_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image4]: ./images/center_2016_12_01_13_30_48_287.jpg "Normal Image"
[image5]: ./images/center_2016_12_01_13_30_48_287_flipped.jpg "Flipped Image"
[image6]: ./model_data/data.png "Training and Validating Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_data/model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 result of the autonomous driving

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_data/model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is little changed model from NVIDIA model. 
See NVIDIA Documentation if you want to know more about it.

https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

/model/NVIDIA_model.py contains comments to explain how the code works.

####2. Attempts to reduce overfitting in the model

Also, model contains l2_regularization in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I designed DenseNet121, and MobileNet for the behavior cloning project, but did not use these one because the training took too long.
I made my own NVIDIA model. And this certainly worked well.

I use dropout for good training.
And this is my model.summary from keras.

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
dropout_1 (Dropout)          (None, 43, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
dropout_2 (Dropout)          (None, 20, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
dropout_4 (Dropout)          (None, 6, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________

####2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of left, center and right lane driving:

![alt text][image1]
![alt text][image2]
![alt text][image3]

these datas are approximately 32000 pictures.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to go back to the center of the road.

these datas are approximately 7000 pictures.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

I finally randomly shuffled the data set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 10.
This image contains my training and validation graph. 

![alt text][image6]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
