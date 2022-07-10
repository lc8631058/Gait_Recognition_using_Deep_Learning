##

---

[//]: # (Image References)
[image1]: ./example/gait_recognition.png
[image2]: ./example/problem-illustration.jpg
[image3]: ./example/problem-illustration2.jpg
[image4]: ./example/GEI_generation.png
[image5]: ./example/incom_GEI_generation.jpg
[image6]: ./example/train_strategy.jpg
[image7]: ./example/End2End_ITCNet.png
[image8]: ./example/workflow.png

# Article: "Person identification from partial gait cycle using fully convolutional neural networks, Neurocomputing", vol. 338, pp. 116–125, 2019, issn: 0925-2312.
25th IEEE International Conference on Image Processing (ICIP) „GAIT RECOGNITION FROM INCOMPLETE GAIT CYCLE“, 2019, pp. 768–772.

### This repository includes part of the codes from my above paper, as the demand of my spuervisor, I will clear up the codes after the conference.

![alt text][image1]

## Problem Formulation 

The Gait Energy Image (GEI) could reflect specific individual’s gait features, using GEI to represent human’s gait features has been a common method for gait recognition. But in some circumstances, the GEI of complete gait cycle is not available, e.g., when the person is walking through a multi-crowd environment or behind some shelters:

![alt text][image2]

In these cases, only a few frames’ gait silhouettes could be extracted to generate the in- complete GEI, which leads to an extremely lower gait recognition rate. This case is not a special case, but can be seen everywhere in railway station, airport and street:

![alt text][image3]

## GEI

Gait Energy Image (GEI) is one kind of gait featur representation, it has been proven to be one of the most simple gait representation and has the best recognition ability:

![alt text][image4]

## Generation of different types of incomplete GEIs

We want to build a network, which can transform different kind of incomplete GEI, whether it's composed of 1 frames or 10 frames of Gait silhouettes or different start frame, direct to complete GEI, whcih has almost one kind of shape for each subject:

![alt text][image5]

## Train strategy

As you can see from this figure, different incomplete GEIs composed of differen number of frames of start frames has various shapes, standing or walking, so it's hard to directly transform them to complete GEI, so we first train many network to achieve a small range transformation, we call it ITCNet, which means Incomplete to Complete Transformation Network, GC means gait cycle, 1/10 GC meams by 1/10 gait cycle composed GEI： 

![alt text][image6]

## Network Structure

Here is the final structure of ITCNet, for more details please refer to my paper “Person Identification form Partial Gait Cycle Using Fully Convolutional Neural Network”, but it still been processed these days by NeuroComputing periodical:

![alt text][image7]

## Workflow

Here is the total workflow of my master thesis:

![alt text][image8]

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
