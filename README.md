---
[//]: # (Image References)
[image1]: ./example/gait recognition.png
---

# Project Title

Gait recognition from incomplete gait-cycle using convolutional neural networks

![alt text][image1]

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

# After activation, use this to open this enviorment every time before you open jupyter notebook 
```
for windows: activate tensorflow
for linux: source activate tensorflow
```

# then open jupyter notebook using this:
```
jupyter notebook
```

### Test model

In `checkpoints_view_invariant` saved all trained models, if you want to test with these models just open any ITCNet.ipynb file, there is a `Test Model` section in the end of it. Because the data is too big, I can't upload them to dropbox. The function `save_decoded` could save the predicted result to any path. The function `get_batches` could split the test data into batches, so you can input any data you want to test to `get_batches`. 


