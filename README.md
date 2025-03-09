# DA6401---Assignment-1

## Fashion-MNIST Image Classifier using Neural Networks

### Introduction
This assignment develop a neural network model using fashion-MNIST dataset.This dataset is used for image classification.This dataset is trained and evaluated using different optimization techniques and hyperparameter tuning.For this code, NumPy, Matplotlib, and Keras libraries are used for building the model and visualizing results. sklearn.model_selection is used for splitting the dataset into  two dataset training and testing.Various optimization algorithms and hyperparameter tuning methods are used to improve the model's performance.

### Dataset
The fashion-MNIST dataset contains 60,000 training images and 10,000 testing images.Each image belongs to one of the 10 fashion items classes. Each image is a 28x28 grayscale image with a total of 784 pixels per image.

The dataset 10 categories are:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

### Data Processing:
Before training the model, each image of the fashion-MNIST dataset is flattened where transform the 28x28 pixel image into a 1D array of length 784. This helps the neural network to process the images as a set of features.Then, the entire training, v testing and validation datasets are normalized to a range between 0 and 1 to improve the performance of the neural network model.Then  use the one hot encoded function to convert the label of each image into one-hot encoded arrays of length 10 where transforming the categorical value into a numerical value.








