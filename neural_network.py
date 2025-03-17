import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import wandb  # Import Weights & Biases for logging experiments

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[128, 64], output_size=10,
                 learning_rate=0.01, optimizer="sgd", weight_init="random",
                 activation="sigmoid", weight_decay=0.0):
        """
        Initialize a flexible feedforward neural network.
        - Supports different weight initialization, activation functions, and optimizers.
        - Includes weight decay (L2 regularization).
        """
        self.layers = [input_size] + hidden_layers + [output_size]  # Define layer sizes
        self.learning_rate = learning_rate  # Learning rate for gradient updates
        self.optimizer = optimizer  # Choose optimization method
        self.weight_init = weight_init  # Weight initialization method
        self.activation = activation  # Activation function type
        self.weight_decay = weight_decay  # L2 regularization strength
        self.init_weights()  # Initialize weights and biases

        # Optimizer-specific parameters
        self.momentum = 0.9  # Momentum factor (for SGD with momentum)
        self.beta = 0.9  # Beta for RMSprop
        self.beta1 = 0.9  # Adam/Nadam first moment decay rate
        self.beta2 = 0.999  # Adam/Nadam second moment decay rate
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.t = 0  # Time step counter for Adam/Nadam

        # Momentum/Nesterov optimizer variables
        self.velocity = [np.zeros_like(w) for w in self.weights]  # Velocity for weights
        self.velocity_bias = [np.zeros_like(b) for b in self.biases]  # Velocity for biases

        # RMSprop optimizer variables
        self.squared_grads = [np.zeros_like(w) for w in self.weights]  # Squared gradients for weights
        self.squared_grads_bias = [np.zeros_like(b) for b in self.biases]  # Squared gradients for biases

        # Adam/Nadam optimizer variables
        self.m = [np.zeros_like(w) for w in self.weights]  # First moment estimate (weights)
        self.v = [np.zeros_like(w) for w in self.weights]  # Second moment estimate (weights)
        self.m_bias = [np.zeros_like(b) for b in self.biases]  # First moment estimate (biases)
        self.v_bias = [np.zeros_like(b) for b in self.biases]  # Second moment estimate (biases)

    def init_weights(self):
        """ Initialize weights and biases based on the chosen method. """
        self.weights = []  # List to store weight matrices
        self.biases = []  # List to store bias vectors
        for i in range(len(self.layers) - 1):
            if self.weight_init == "random":
                # Initialize with small random values
                self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.01)
            elif self.weight_init == "xavier":
                # Xavier initialization for better weight scaling
                self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(1 / self.layers[i]))
            # Initialize biases as zeros
            self.biases.append(np.zeros((1, self.layers[i+1])))

    def activation_function(self, z):
        """ Compute activation function output based on user choice. """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -10, 10)))  # Sigmoid function with clipping to prevent overflow
        elif self.activation == "tanh":
            return np.tanh(z)  # Tanh activation function
        elif self.activation == "relu":
            return np.maximum(0, z)  # ReLU activation function

    def activation_derivative(self, a):
        """ Compute the derivative of the chosen activation function. """
        if self.activation == "sigmoid":
            return a * (1 - a)  # Derivative of sigmoid
        elif self.activation == "tanh":
            return 1 - a**2  # Derivative of tanh
        elif self.activation == "relu":
            return (a > 0).astype(float)  # Derivative of ReLU (1 for positive, 0 for negative)

    def softmax(self, z):
        """ Compute softmax function for output layer. """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Normalize to get probabilities

    def forward(self, X):
        """ Perform forward propagation through the network. """
        activations = [X]  # Store activations (input layer)
        z_values = []  # Store weighted sums (Z values)

        # Iterate through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]  # Compute weighted sum
            z_values.append(z)  # Store weighted sum
            activations.append(self.activation_function(z))  # Apply activation function

        # Compute final layer (output layer)
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]  # Compute weighted sum
        z_values.append(z_out)  # Store output layer Z values
        activations.append(self.softmax(z_out))  # Apply softmax for probability output

        return activations, z_values  # Return activations and Z values

    def compute_loss(self, y_true, y_pred):
        """ Compute cross-entropy loss with optional L2 regularization. """
        loss = -np.sum(y_true * np.log(y_pred + self.epsilon)) / y_true.shape[0]  # Compute cross-entropy loss
        l2_penalty = self.weight_decay * sum(np.sum(w**2) for w in self.weights) / 2  # Compute L2 regularization term
        return loss + l2_penalty  # Return total loss

    def compute_accuracy(self, y_true, y_pred):
        """ Compute accuracy by comparing true labels and predicted labels. """
        correct_predictions = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))  # Count correct matches
        return (correct_predictions / y_true.shape[0]) * 100  # Compute percentage accuracy

    def backward(self, X, y_true, activations, z_values):
        """ Backpropagation to compute gradients for weight and bias updates. """
        
        # Initialize gradients for weights and biases with zeros
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
    
        # Compute gradient for the output layer
        dL_dz = activations[-1] - y_true  # Compute derivative of loss with respect to output layer activation
        gradients_w[-1] = np.dot(activations[-2].T, dL_dz) + self.weight_decay * self.weights[-1]  # Weight gradient
        gradients_b[-1] = np.sum(dL_dz, axis=0, keepdims=True)  # Bias gradient
    
        # Compute gradients for hidden layers (moving backward)
        for i in reversed(range(len(self.weights) - 1)):  # Iterate from last hidden layer to first
            dL_dz = np.dot(dL_dz, self.weights[i+1].T) * self.activation_derivative(activations[i+1])  # Backpropagate error
            gradients_w[i] = np.dot(activations[i].T, dL_dz) + self.weight_decay * self.weights[i]  # Compute weight gradient
            gradients_b[i] = np.sum(dL_dz, axis=0, keepdims=True)  # Compute bias gradient
    
        return gradients_w, gradients_b  # Return computed gradients


    def update_weights(self, gradients_w, gradients_b):
        """ Apply gradient updates using different optimizers. """
        
        self.t += 1  # Increment time step (used in Adam/Nadam for bias correction)
    
        for i in range(len(self.weights)):  # Loop through each layer
            if self.optimizer == "sgd":  # Standard Stochastic Gradient Descent (SGD)
                self.weights[i] -= self.learning_rate * gradients_w[i]  # Update weights
                self.biases[i] -= self.learning_rate * gradients_b[i]  # Update biases
    
            elif self.optimizer == "momentum":  # SGD with momentum
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]  # Compute velocity update
                self.weights[i] += self.velocity[i]  # Apply velocity update to weights
                self.biases[i] -= self.learning_rate * gradients_b[i]  # Update biases
    
            elif self.optimizer == "nesterov":  # Nesterov Accelerated Gradient (NAG)
                temp_weights = self.weights[i] + self.momentum * self.velocity[i]  # Lookahead update for weights
                temp_biases = self.biases[i] + self.momentum * self.velocity_bias[i]  # Lookahead update for biases
    
                # Compute velocity update
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients_w[i]
                self.velocity_bias[i] = self.momentum * self.velocity_bias[i] - self.learning_rate * gradients_b[i]
    
                # Apply final update
                self.weights[i] = temp_weights + self.velocity[i]
                self.biases[i] = temp_biases + self.velocity_bias[i]
    
            elif self.optimizer == "rmsprop":  # RMSprop optimizer
                # Update squared gradient moving average for weights
                print('selfbeta', self.beta)
                self.squared_grads[i] = self.beta * self.squared_grads[i] + (1 - self.beta) * (gradients_w[i] ** 2)
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.squared_grads[i]) + self.epsilon)
    
                # Update squared gradient moving average for biases
                self.squared_grads_bias[i] = 0.9 * self.squared_grads_bias[i] + 0.1 * (gradients_b[i] ** 2)
                self.biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.squared_grads_bias[i]) + self.epsilon)
    
            elif self.optimizer == "adam":  # Adam optimizer (Adaptive Moment Estimation)
                # Compute biased first moment (momentum) update for weights
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients_w[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
    
                # Bias correction for first and second moments
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
    
                # Weight update
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
                # Compute biased first moment update for biases
                self.m_bias[i] = self.beta1 * self.m_bias[i] + (1 - self.beta1) * gradients_b[i]
                self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (gradients_b[i] ** 2)
    
                # Bias correction for biases
                m_hat_bias = self.m_bias[i] / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v_bias[i] / (1 - self.beta2 ** self.t)
    
                # Bias update
                self.biases[i] -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
    
            elif self.optimizer == "nadam":  # Nadam optimizer (Adam with Nesterov acceleration)
                # Compute first moment update for weights
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients_w[i]
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
    
                # Bias correction for first and second moments
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
    
                # Apply Nadam update formula for weights
                self.weights[i] -= self.learning_rate * (self.momentum * m_hat + (1 - self.momentum) * gradients_w[i]) / (np.sqrt(v_hat) + self.epsilon)
    
                # Compute first moment update for biases
                self.m_bias[i] = self.beta1 * self.m_bias[i] + (1 - self.beta1) * gradients_b[i]
                self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (gradients_b[i] ** 2)
    
                # Bias correction for biases
                m_hat_bias = self.m_bias[i] / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v_bias[i] / (1 - self.beta2 ** self.t)
    
                # Apply Nadam update formula for biases
                self.biases[i] -= self.learning_rate * (self.momentum * m_hat_bias + (1 - self.momentum) * gradients_b[i]) / (np.sqrt(v_hat_bias) + self.epsilon)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """ Train the network using mini-batch gradient descent and track accuracy. """
    
        num_samples = X_train.shape[0]  # Get the total number of training samples
    
        for epoch in range(epochs):  # Loop through the specified number of epochs
            # Shuffle the training data to ensure randomness in batches
            indices = np.arange(num_samples)  # Create an array of indices
            np.random.shuffle(indices)  # Shuffle the indices randomly
            X_train, y_train = X_train[indices], y_train[indices]  # Reorder training data accordingly
    
            # Mini-batch training
            for i in range(0, num_samples, batch_size):  # Iterate over data in batches
                X_batch = X_train[i:i + batch_size]  # Extract mini-batch of inputs
                y_batch = y_train[i:i + batch_size]  # Extract corresponding mini-batch of labels
    
                # Forward pass (compute predictions)
                activations, z_values = self.forward(X_batch)
    
                # Compute gradients using backpropagation
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, z_values)
    
                # Update weights and biases using the chosen optimizer
                self.update_weights(gradients_w, gradients_b)
    
            # Compute training loss & accuracy after one epoch
            train_activations, _ = self.forward(X_train)  # Forward pass for full training data
            Train_loss = self.compute_loss(y_train, train_activations[-1])  # Compute loss
            Train_accuracy = self.compute_accuracy(y_train, train_activations[-1])  # Compute accuracy
    
            # Compute validation loss & accuracy after one epoch
            val_activations, _ = self.forward(X_val)  # Forward pass for validation data
            Val_loss = self.compute_loss(y_val, val_activations[-1])  # Compute validation loss
            Val_accuracy = self.compute_accuracy(y_val, val_activations[-1])  # Compute validation accuracy
    
            # Log training and validation metrics to Weights & Biases (WandB)
            wandb.log({'Train_loss': Train_loss})  # Log training loss
            wandb.log({'Train_accuracy': Train_accuracy})  # Log training accuracy
            wandb.log({'epoch': epoch + 1})  # Log current epoch
            wandb.log({'Val_loss': Val_loss})  # Log validation loss
            wandb.log({'Val_accuracy': Val_accuracy})  # Log validation accuracy
    
            # Print training progress for each epoch
            print(f"Epoch {epoch+1}: Train Loss = {Train_loss:.4f}, Train Acc = {Train_accuracy:.4f}, Val Loss = {Val_loss:.4f}, Val Acc = {Val_accuracy:.4f}")
