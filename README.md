# DA6401---Assignment-1

## Fashion-MNIST Image Classifier using Neural Networks

### Introduction
This assignment develop a neural network model using fashion-MNIST dataset.This dataset is used for image classification.This dataset is trained and evaluated using different optimization techniques and hyperparameter tuning.

For this code, NumPy, Matplotlib, and Keras libraries are used for building the model and visualizing results. 

naive method is used for splitting the dataset into  two dataset training and testing.Various optimization algorithms and hyperparameter tuning methods are used to improve the model's performance.

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
Before training the model, each image of the fashion-MNIST dataset is flattened where transform the 28x28 pixel image into a 1D array of length 784. This helps the neural network to process the images as a set of features.

Then, the entire training, testing and validation datasets are normalized to a range between 0 and 1 to improve the performance of the neural network model.

Then  apply one hot encoded function to convert the label of each image into one-hot encoded arrays of length 10 where transforming the categorical value into a numerical value.

### Optimizers:
The following optimizers are used:

1.Stochastic Gradient Descent (SGD)

2.Momentum-based Gradient Descent

3.Nesterov Accelerated Gradient

4.RMSprop

5.Adam
6.Nadam

### Training Process
#### 1.initialization:
The neural network is initialized with the parameters before training: 

input size (784 for Fashion-MNIST)

output size (10 classes)

number of hidden layers

nodes in each hidden layer

weight initialization.

#### 2.Forward Propagation
This step data is passed through the neural network where each hidden layer uses an activation function  and the output layer uses to predict class probabilities.

#### 3.Backward Propagation
Cross-entropy loss is calculated, and gradients are computed using backpropagation.Optimizers like SGD, Momentum, and Nadam update the network’s parameters over several epochs.Training loss, accuracy, and validation accuracy are monitored to evaluate the model's performance.

### Hyperparameters:
Using the sweep functionality provided by wandb to finding the best values for the hyperparameters listed below:

number of epochs: 5, 10

number of hidden layers: 3, 4, 5

size of every hidden layer: 32, 64, 128

weight decay (L2 regularisation): 0, 0.0005, 0.5

learning rate: 1e-3, 1 e-4

optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam

batch size: 16, 32, 64

weight initialisation: random, Xavier

activation functions: sigmoid, tanh, ReLU

### Confusion Matrix:

After finding the best hyperparameters, evaluate test accuracy and plot the confusion matrix for true vs predicted labels.

### Prediction:
The model was evaluated on the Fashion-MNIST dataset using three different hyperparameter configurations. The test accuracy was measured for each configuration, and results were logged with Wandb.

Wandb Project Report: [https://wandb.ai/ma23m011-iit-madras/DA6401_Assignment1_ma23m011/reportlist]






