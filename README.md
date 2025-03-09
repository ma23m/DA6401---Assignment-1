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
Before training the model, each image is flattened, transforming the 28x28 pixel matrix into a 1D array of length 784. This allows the neural network to process the images as a set of features. Furthermore, the pixel values across the entire training, validation, and test datasets are normalized to a range between 0 and 1 to improve the stability and performance of the model.

Additionally, the labels for each image are converted into one-hot encoded arrays of length 10, where each array corresponds to one of the 10 possible classes, transforming the categorical labels into a numerical format suitable for training the neural network.








