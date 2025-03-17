import numpy as np  # Import NumPy for numerical operations
from keras.datasets import mnist, fashion_mnist  # Import datasets from Keras

def one_hot_encode(y, num_classes=10):
    """ Convert labels into one-hot encoded format """
    one_hot = np.zeros((y.size, num_classes))  # Create a zero matrix of size (number of labels, num_classes)
    one_hot[np.arange(y.size), y] = 1  # Set the appropriate index to 1 for each label
    return one_hot  # Return the one-hot encoded labels

def load_dataset(dataset_name):
    """ Load and preprocess the chosen dataset """
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load MNIST dataset
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # Load Fashion MNIST dataset

    # Normalize pixel values to range [0, 1] and flatten the images into 1D vectors
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0  

    # Create a validation set using the last 6000 samples from the training set
    x_val = x_train[54000:]  
    y_val = y_train[54000:]  
    
    # Keep the first 54,000 samples for training
    x_train = x_train[:54000]  
    y_train = y_train[:54000]  

    # Convert labels to one-hot encoded format
    y_train = one_hot_encode(y_train)  
    y_val = one_hot_encode(y_val)  
    y_test = one_hot_encode(y_test)  

    return x_train, y_train, x_val, y_val, x_test, y_test  # Return the processed dataset
