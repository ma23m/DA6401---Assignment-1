import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import wandb  # Import Weights & Biases for experiment tracking
from neural_network import NeuralNetwork  # Import base NeuralNetwork class
from dataset_loader import load_dataset  # Import dataset loading function

# Define Modified Neural Network
class ModifiedNeuralNetwork(NeuralNetwork):
    def __init__(self, loss_function="cross_entropy", **kwargs):
        """
        Extends NeuralNetwork to support both Cross-Entropy and Squared Error Loss.
        """
        super().__init__(**kwargs)  # Call the parent class (NeuralNetwork) initializer
        self.loss_function = loss_function  # Store the selected loss function type

    def compute_loss(self, y_true, y_pred):
        """ Compute loss based on selected loss function. """
        if self.loss_function == "cross_entropy":
            # Compute Cross-Entropy Loss (for classification)
            return -np.sum(y_true * np.log(y_pred + self.epsilon)) / y_true.shape[0]
        elif self.loss_function == "squared_error":
            # Compute Mean Squared Error Loss (for regression-like tasks)
            return np.mean((y_true - y_pred) ** 2)

    def compute_accuracy(self, y_true, y_pred):
        """ Compute accuracy by comparing true vs predicted labels. """
        # Get the predicted class (index of max probability) and compare with the true labels
        correct_predictions = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        return (correct_predictions / y_true.shape[0]) * 100  # Convert to percentage

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """ Train the network and track loss & accuracy per epoch. """
        num_samples = X_train.shape[0]  # Get the total number of training samples
        loss_history = []  # Store loss values for each epoch
        accuracy_history = []  # Store accuracy values for each epoch

        for epoch in range(epochs):  # Loop through each epoch
            # Shuffle training data to ensure randomness in mini-batches
            indices = np.arange(num_samples)  # Create an array of indices
            np.random.shuffle(indices)  # Shuffle indices randomly
            X_train, y_train = X_train[indices], y_train[indices]  # Apply shuffled order to data

            for i in range(0, num_samples, batch_size):  # Iterate through mini-batches
                X_batch = X_train[i:i + batch_size]  # Extract mini-batch of input features
                y_batch = y_train[i:i + batch_size]  # Extract mini-batch of labels

                # Forward pass (compute predictions)
                activations, z_values = self.forward(X_batch)

                # Compute gradients using backpropagation
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, z_values)

                # Update weights using selected optimization method
                self.update_weights(gradients_w, gradients_b)

            # Compute validation loss & accuracy at the end of each epoch
            val_activations, _ = self.forward(X_val)  # Forward pass on validation set
            val_loss = self.compute_loss(y_val, val_activations[-1])  # Compute validation loss
            val_accuracy = self.compute_accuracy(y_val, val_activations[-1])  # Compute validation accuracy

            # Store loss and accuracy for later visualization
            loss_history.append(val_loss)
            accuracy_history.append(val_accuracy)

            # Print progress for the current epoch
            print(f"Epoch {epoch+1}: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")

            # Log loss & accuracy per epoch in Weights & Biases (WandB)
            wandb.log({
                f"{self.loss_function} Loss": val_loss,  # Log loss
                f"{self.loss_function} Accuracy": val_accuracy,  # Log accuracy
                "Epoch": epoch  # Log current epoch number
            })

        return loss_history, accuracy_history  # Return stored loss & accuracy history for plotting
