# -*- coding: utf-8 -*-
"""DL_Assignment_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HUW7QEPPY_ZCcMvCHtCDttsTqJHoBvG-

DA6401 Assignment-1

NAME: MOUSINA BARMAN

ROLL: MA23M011
"""

#For niumerical operation import numpy
import numpy as np
#For visualization import Matplotlib
import matplotlib.pyplot as plt
#For logging import Weights & Biases
import wandb
#import for fashion-mnist dataset from keras
from keras.datasets import fashion_mnist

# Initialize wandb for logging
wandb.init(project="DA6401_Assignment1_ma23m011", name="sample-images-1")

# Load fashion-mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Class labels define based on fashion-mnist categories
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Initialize empty list to store sample  image and their labels
sample_images = []
sample_labels = []

#use for loop to add sample image and their class
for class_id in range(10):
    idx = np.where(y_train == class_id)[0][0] # Find the first index where y_train and class label equal
    sample_images.append(x_train[idx]) # Add the image to the list
    sample_labels.append(class_labels[class_id]) # Add the label to the list

# With caption log images to wandb
wandb.log({"Sample Images": [wandb.Image(img, caption=label) for img, label in zip(sample_images, sample_labels)]})


fig, axes = plt.subplots(2, 5, figsize=(10, 5)) # plot 2x5 grid image
fig.suptitle("Sample Images from Fashion-MNIST", fontsize=14)# Add a title

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray') # Display image in grayscale
    ax.set_title(sample_labels[i]) # Set the title as the class label
    ax.axis("off") # Hide axes

# Show the plotted images
plt.show()

# Finish the wandb logging run
wandb.finish()

import numpy as np
#import for fashion-mnist dataset from keras
from keras.datasets import fashion_mnist

# Load fashion-mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize  0 to 1 pixel values and flatten images convert 1D vector
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# From training dataset 60000 sample images create validation set
x_val = x_train[54000:]
y_val = y_train[54000:]

# For training using 54000 images
x_train = x_train[:54000]
y_train = y_train[:54000]

# Convert labels into one-hot encoding
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))# Create a zero matrix
    one_hot[np.arange(y.size), y] = 1 # Set the corresponding class index to 1
    return one_hot

#training,test and validation labels apply one hot encoding
y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)
y_test = one_hot_encode(y_test)

#Define Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[128, 64], output_size=10,
                 learning_rate=0.01, optimizer="sgd", weight_init="random",
                 activation="sigmoid", weight_decay=0.0):
        #initialize parameters for feedforward neural network
        self.layers = [input_size] + hidden_layers + [output_size]
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.activation = activation
        self.weight_decay = weight_decay
        self.init_weights()

        #For different optimizer technique use optimizer-specific variables
        self.momentum = 0.9 #mumentum parameter
        self.beta1 = 0.9  # Adam/Nadam parameter
        self.beta2 = 0.999 # Adam/Nadam parameter
        self.epsilon = 1e-8
        self.velocity = [np.zeros_like(w) for w in self.weights] #mumentum velocity
        self.squared_grads = [np.zeros_like(w) for w in self.weights]
        self.m = [np.zeros_like(w) for w in self.weights] #adam 1st moment
        self.v = [np.zeros_like(w) for w in self.weights] #adam 2nd moment
        self.t = 0  # Time step

    def init_weights(self):
        # Initialize weights and biases
        self.weights = [] #Empty list define to store weight matrix
        self.biases = [] #Empty list define to store bias vector
        #loop use for random and xavier initialization
        for i in range(len(self.layers) - 1):
            if self.weight_init == "random":
                self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.01)
            elif self.weight_init == "xavier":
                self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i+1])))

    # Calculate activation function value based user expectation.
    def activation_function(self, z):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -10, 10)))
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0, z)

    #calculate activation function derivative
    def activation_derivative(self, a):
        if self.activation == "sigmoid":
            return a * (1 - a)
        elif self.activation == "tanh":
            return 1 - a**2
        elif self.activation == "relu":
            return (a > 0).astype(float)

    #For output layer calculate softmax function value
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    #Forward propagation
    def forward(self, X):
        #store each layer activation value
        activations = [X]
        #store weighted sum value
        z_values = []

         #Use for loop for calculate weighted sum and  apply activation function
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            activations.append(self.activation_function(z))

        #Final layer computation using softmax function
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z_out)
        activations.append(self.softmax(z_out))

        return activations, z_values

    #Cross entropy loss computation with L2 ragularization
    def compute_loss(self, y_true, y_pred):
        #cross entropy loss
        loss = -np.sum(y_true * np.log(y_pred + self.epsilon)) / y_true.shape[0]
        #L2 regularization
        l2_penalty = self.weight_decay * sum(np.sum(w**2) for w in self.weights) / 2
        return loss + l2_penalty

    #Compute accuracy by comparing true vs predicted labels.
    def compute_accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        return correct_predictions / y_true.shape[0]

     #Use backpropagation to calculate gradients.
    def backward(self, X, y_true, activations, z_values):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]

        #Compute output layer gradient
        dL_dz = activations[-1] - y_true
        gradients_w[-1] = np.dot(activations[-2].T, dL_dz) + self.weight_decay * self.weights[-1]
        gradients_b[-1] = np.sum(dL_dz, axis=0, keepdims=True)

        # Compute Hidden layers gradient
        for i in reversed(range(len(self.weights) - 1)):
            dL_dz = np.dot(dL_dz, self.weights[i+1].T) * self.activation_derivative(activations[i+1])
            gradients_w[i] = np.dot(activations[i].T, dL_dz) + self.weight_decay * self.weights[i]
            gradients_b[i] = np.sum(dL_dz, axis=0, keepdims=True)

        return gradients_w, gradients_b

    # Apply gradient updates using different optimizers.
    def update_weights(self, gradients_w, gradients_b):
        self.t += 1  # Update time step for Adam/Nadam

        for i in range(len(self.weights)):
            # Apply stochastic gradient descent
            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            #Apply momentum gradient descent
            elif self.optimizer == "momentum":
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]
                self.weights[i] += self.velocity[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            #Apply nestrov
            elif self.optimizer == "nesterov":
                temp_weights = self.weights[i] + self.momentum * self.velocity[i]
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]
                self.weights[i] = temp_weights + self.velocity[i]

            #Apply rmsprop optimizer
            elif self.optimizer == "rmsprop":
                self.squared_grads[i] = 0.9 * self.squared_grads[i] + 0.1 * (gradients_w[i] ** 2)
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.squared_grads[i]) + self.epsilon)

            #Apply adam optimizer
            elif self.optimizer == "adam":
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients_w[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            #Apply nadam optimizer
            elif self.optimizer == "nadam":
                m_hat = (self.beta1 * self.m[i] + (1 - self.beta1) * gradients_w[i]) / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                self.weights[i] -= self.learning_rate * (self.momentum * m_hat + (1 - self.momentum) * gradients_w[i]) / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        #Train nural network using mini-batch gradient descent.
        num_samples = X_train.shape[0]

        #use for loop for shuffle training data
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                #Use forward pass
                activations, z_values = self.forward(X_batch)

                # Compute gradients of weight and bias using backpropagation
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, z_values)

                # Update weights and bias
                self.update_weights(gradients_w, gradients_b)

            # Compute training loss & accuracy
            train_activations, _ = self.forward(X_train)
            Train_loss = self.compute_loss(y_train, train_activations[-1])
            Train_accuracy = compute_accuracy(y_train, train_activations[-1])

            # Compute validation loss & accuracy
            val_activations, _ = self.forward(X_val)
            Val_loss = self.compute_loss(y_val, val_activations[-1])
            Val_accuracy = compute_accuracy(y_val, val_activations[-1])
            wandb.log({'Train_loss': Train_loss})
            wandb.log({'Train_accuracy': Train_accuracy })
            wandb.log({'epoch': epoch + 1})
            wandb.log({'Val_loss': Val_loss})
            wandb.log({'Val_accuracy': Val_accuracy })

            print(f"Epoch {epoch+1}: Train Loss = {Train_loss:.4f}, Train Acc = {Train_accuracy:.4f}, Val Loss = {Val_loss:.4f}, Val Acc = {Val_accuracy:.4f}")

def compute_accuracy(y_true, y_pred):
        #Compute accuracy by comparing true vs predicted labels.
        correct_predictions = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy*100

# Create a neural network with a flexible architecture
nn = NeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10)

# Forward pass example
sample_input = x_train[:50]  # Take 5 sample images
# print(sample_input.shape)
output_probs = nn.forward(sample_input)[0]  # Get output probability distribution

# Print predictions
#print("Predicted class probabilities:\n", output_probs)
print("Predicted classes:", np.argmax(output_probs[3], axis=1))

# # Create Neural Network
# nn = NeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10, learning_rate=0.01, optimizer="sgd", weight_init="xavier", activation="relu", weight_decay=0.0005)

# # Train the model and track accuracy
# nn.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=32)

!pip install wandb

import wandb
import numpy as np
#For easy attribute access import SimpleNamespace
from types import SimpleNamespace
# import Random  for randomization
import random

key = input('Enter your API:')
wandb.login(key=key)

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',# Use Bayesian optimization
    'name' : 'sweep cross entropy-14', #sweep name
    'metric': {
      'name': 'Val_accuracy',
      'goal': 'maximize' # maximize validation accuracy
    },
    'parameters': {
        'epochs': {
            'values': [5,10] #number of training epoch
        },
        'hidden_layers':{
            'values':[3,4,5] #number of hidden layer
        },
         'hidden_size':{
            'values':[32,64,128] #hidden layer size
        },
        'weight_decay':{
            'values':[0, 0.0005, 0.5] # Regularization
        },
        'learning_rate': {
            'values': [1e-3, 1e-4] # Learning rate
        },
        'optimizer': {
            'values': ['rmsprop', 'nadam','adam', 'nag','mgd','sgd'] # Optimizers
        },
        'batch_size':{
            'values':[16,32,64] # Batch sizes for training
        },
        'weight_init': {
            'values': ['xavier','random'] # Weight initialization
        },
        'activation': {
            'values': ['relu','tanh','sigmoid'] # Activation function
        },
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='DA6401_Assignment1_ma23m011')

def main():
    with wandb.init() as run:
      # Generate a unique run name based on the hyperparameters
        run_name="-ac_"+wandb.config.activation+"-hs_"+str(wandb.config.hidden_size)+"-epc_"+str(wandb.config.epochs)+"-hl_"+str(wandb.config.hidden_layers)+"-regu_"+str(wandb.config.weight_decay)+"-eta_"+str(wandb.config.learning_rate)+"-optmz_"+wandb.config.optimizer+"-batch_"+str(wandb.config.batch_size)+"-wght_"+wandb.config.weight_init
        wandb.run.name=run_name
        # Create a Neural Network with the selected hyperparameters
        nn = NeuralNetwork(input_size=784, hidden_layers=[wandb.config.hidden_size] * wandb.config.hidden_layers, output_size=10, learning_rate=wandb.config.learning_rate, optimizer=wandb.config.optimizer, weight_init=wandb.config.weight_init, activation=wandb.config.activation, weight_decay=wandb.config.weight_decay)
        nn.train(x_train, y_train, x_val, y_val, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size)


# # Create Neural Network
# nn = NeuralNetwork(input_size=784, hidden_layers=[128, 64], output_size=10, learning_rate=0.01, optimizer="sgd")

# # Train the model and track accuracy
# nn.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=32)

wandb.agent(sweep_id, function=main,count=100) # calls main function for count number of times.
# Finish the W&B run
wandb.finish()

import wandb
#-ac_relu-hs_128-epc_10-hl_5-regu_0.0005-eta_0.001-optmz_sgd-batch_64-wght_xavier
best_config = {
    'hidden_layers': [128]*5,
    'learning_rate': 0.001,
    'optimizer': 'sgd',
    'weight_init': 'xavier',
    'activation': 'relu',
    'weight_decay': 0.0005,
    'epochs': 10,
    'batch_size': 64
}

# Define a Sweep for Confusion Matrix
sweep_config_conf_matrix = {
    'method': 'grid',
    'name': 'Confusion Matrix Sweep-2',
    'parameters': {
        'model_name': {'values': ['best_model']}
    }
}

# Create a sweep for confusion matrix
sweep_id_conf_matrix = wandb.sweep(sweep=sweep_config_conf_matrix, project='DA6401_Assignment1_ma23m011')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

def train_best_model_and_plot_confusion_matrix():
    wandb.init(project="DA6401_Assignment1_ma23m011", name="Best Model - Conf Matrix")

    # Train Best Model
    best_model = NeuralNetwork(
        input_size=784,
        hidden_layers=best_config['hidden_layers'],
        output_size=10,
        learning_rate=best_config['learning_rate'],
        optimizer=best_config['optimizer'],
        weight_init=best_config['weight_init'],
        activation=best_config['activation'],
        weight_decay=best_config['weight_decay']
    )

    num_epochs = best_config['epochs']
    batch_size = best_config['batch_size']

    # Store test accuracy for plotting
    test_accuracies = []

    for epoch in range(num_epochs):
        best_model.train(x_train, y_train, x_val, y_val, epochs=1, batch_size=batch_size)

        # Forward pass to get predictions on test set
        y_pred_probs = best_model.forward(x_test)[0][-1]  # Get softmax output
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Compute test accuracy
        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
        test_accuracies.append(accuracy * 100)  # Convert to percentage

        print(f"Epoch {epoch+1}: Test Accuracy = {accuracy*100:.2f}%")

        # Log accuracy per epoch to WandB
        wandb.log({"Test Accuracy": accuracy * 100, "Epoch": epoch + 1})

    # Plot Test Accuracy Over Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), test_accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Over Epochs")
    plt.grid()

    # Log the accuracy plot
    wandb.log({"Test Accuracy Plot": wandb.Image(plt)})
    plt.show()

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Final Test Accuracy: {accuracy*100:.2f}%")

    # Log confusion matrix to WandB
    wandb.log({"Confusion Matrix": wandb.Image(fig)})

    plt.show()

# Run the Sweep
wandb.agent(sweep_id_conf_matrix, function=train_best_model_and_plot_confusion_matrix)

# Define Modified Neural Network
class ModifiedNeuralNetwork(NeuralNetwork):
    def __init__(self, loss_function="cross_entropy", **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function

    def compute_loss(self, y_true, y_pred):
        # Compute loss based on loss function.
        if self.loss_function == "cross_entropy":
            return -np.sum(y_true * np.log(y_pred + self.epsilon)) / y_true.shape[0]
        elif self.loss_function == "squared_error":
            return np.mean((y_true - y_pred) ** 2)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        #Train the neural network using mini-batch gradient descent
        num_samples = X_train.shape[0]
        loss_history = []  # Store loss values per epoch

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                #Use forward pass
                activations, z_values = self.forward(X_batch)

                # Compute gradients
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, z_values)

                # Update weights
                self.update_weights(gradients_w, gradients_b)

            # Compute training loss
            train_activations, _ = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_activations[-1])
            loss_history.append(train_loss)

            print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")

            # Log loss per epoch in WandB
            wandb.log({f"{self.loss_function} Loss": train_loss, "Epoch": epoch})

        return loss_history  # Return loss values for plotting

import numpy as np
import matplotlib.pyplot as plt
import wandb

# Initialize WandB
wandb.init(project="DA6401_Assignment1_ma23m011", name="loss_comparison")


#Train with Cross-Entropy Loss
cross_entropy_model = ModifiedNeuralNetwork(
    input_size=784, hidden_layers=[128]*5, output_size=10,
    learning_rate=0.001, optimizer="sgd", weight_init="xavier",
    activation="relu", weight_decay=0.0005, loss_function="cross_entropy"
)

cross_entropy_losses = cross_entropy_model.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=64)

# Train with Squared Error Loss
squared_error_model = ModifiedNeuralNetwork(
    input_size=784, hidden_layers=[128]*5, output_size=10,
    learning_rate=0.001, optimizer="sgd", weight_init="xavier",
    activation="relu", weight_decay=0.0005, loss_function="squared_error"
)

squared_error_losses = squared_error_model.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=64)

# Plot the Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(cross_entropy_losses, label="Cross-Entropy Loss", color="blue")
plt.plot(squared_error_losses, label="Squared Error Loss", color="red", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Cross-Entropy vs. Squared Error Loss")
plt.legend()
plt.grid()

# Log the loss plot to WandB
wandb.log({"Loss Comparison Plot": wandb.Image(plt)})
plt.show()

import wandb
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and flatten the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

#Train and Evaluate Function
def train_and_evaluate(config, name):
    wandb.init(project="DA6401_Assignment1_ma23m011", name=name, reinit=True)

    # Train the model with given config
    model = ModifiedNeuralNetwork(
        input_size=784, hidden_layers=config['hidden_layers'],
        output_size=10, learning_rate=config['learning_rate'],
        optimizer=config['optimizer'], weight_init=config['weight_init'],
        activation=config['activation'], weight_decay=config['weight_decay']
    )

    num_epochs = 10
    loss_values = []  # store loss per epoch
    test_accuracies = []  # store test accuracy per epoch

    for epoch in range(num_epochs):
        train_loss = model.train(x_train, y_train, x_test, y_test, epochs=1, batch_size=config['batch_size'])
        loss_values.append(train_loss[-1])  # Store last loss of the epoch

        # Compute test accuracy
        y_pred_probs = model.forward(x_test)[0][-1]
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]
        test_accuracies.append(accuracy * 100)

        # Log to WandB
        wandb.log({"Epoch": epoch + 1, "Test Accuracy": accuracy * 100, f"{name} Loss": train_loss[-1]})

    # Plot Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{name} Loss Over Epochs")
    plt.grid()
    wandb.log({f"{name} Loss Plot": wandb.Image(plt)})
    plt.close()

    # Plot Test Accuracy Curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', linestyle='-', color='g')
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"{name} Test Accuracy Over Epochs")
    plt.grid()
    wandb.log({f"{name} Accuracy Plot": wandb.Image(plt)})
    plt.close()

    print(f"{name} - Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    return test_accuracies[-1]

# Run 3 Best Configurations based on fshoin mnist accuracy
configs = [
    {"hidden_layers": [128]*5, "activation": "relu", "optimizer": "sgd", "learning_rate": 0.001, "batch_size": 64, "weight_decay": 0.0005, "weight_init": "xavier"},
    {"hidden_layers": [64]*3, "activation": "sigmoid", "optimizer": "adam", "learning_rate": 0.001, "batch_size": 64, "weight_decay": 0.0005, "weight_init": "xavier"},
    {"hidden_layers": [128]*5, "activation": "tanh", "optimizer": "rmsprop", "learning_rate": 0.001, "batch_size":64, "weight_decay": 0.0005, "weight_init": "xavier"}
]

results = []
for i, config in enumerate(configs):
    accuracy = train_and_evaluate(config, f"Config {i+1} Final")
    results.append((f"Config {i+1}", accuracy))

# Print Final Results
print("\nFinal Results on MNIST:")
for config_name, acc in results:
    print(f"{config_name}: {acc:.2f}%")

