##

---

**Vision based Vehicle Detection and Tracking using Machine Learning and HOG**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector, to compare which kind of combination could lead to the best result
* Note: for those first two steps don't forget to normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./example/gait_recognition.png
[image2]: ./example/gait_recognition.png


# Project Title

Gait recognition from incomplete gait-cycle using convolutional neural networks

![alt text][image1]

### Problem Formulation 

The Gait Energy Image (GEI) could reflect specific individual’s gait features, using GEI to represent human’s gait features has been a common method for gait recognition. But in some circumstances, the GEI of complete gait cycle is not available, e.g., when the person is walking through a multi-crowd environment or behind some shelters:

![alt text][image2]

In these cases, only a few frames’ gait silhouettes could be extracted to generate the in- complete GEI, which leads to an extremely lower gait recognition rate.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software

```
install anaconda on your computer
```

### Installing

A step by step series of examples that tell you have to get a development env running

For windows:
```
open cmd
```

For mac os/ linux:
```
open terminal
```

cd to the path that the `tensorflow.yml` file exist

#For other person to use the environment, first use this to activate the tensorflow.yml:
```
conda env create -f tensorflow.yml
```

#### After activation, use this to open this enviorment every time before you open jupyter notebook 
```
for windows: activate tensorflow
for linux: source activate tensorflow
```

#### then open jupyter notebook using this:
```
jupyter notebook
```

### Test model

In `checkpoints_view_invariant` saved all trained models, if you want to test with these models just open any ITCNet.ipynb file, there is a `Test Model` section in the end of it. Because the data is too big, I can't upload them to dropbox. The function `save_decoded` could save the predicted result to any path. The function `get_batches` could split the test data into batches, so you can input any data you want to test to `get_batches`. 
