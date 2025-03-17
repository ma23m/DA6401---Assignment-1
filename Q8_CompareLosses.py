import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import wandb  # Import Weights & Biases for experiment tracking
from neural_network import NeuralNetwork  # Import base NeuralNetwork class
from dataset_loader import load_dataset  # Import dataset loading function
from ModifiedNN_ForQ8_Q10 import ModifiedNeuralNetwork  # Import modified NN with multiple loss functions

# Initialize WandB 
wandb.init(project="DA6401_Assignment1_ma23m011", name="loss_accuracy_comparison")  # Initialize WandB for logging

# best config:
# -ac_tanh-hs_128-epc_10-hl_4-regu_0.0005-eta_0.001-optmz_nadam-batch_64-wght_xavier

# Load Dataset 
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("fashion_mnist")  # Load Fashion-MNIST dataset

# Train with Cross-Entropy Loss 
cross_entropy_model = ModifiedNeuralNetwork(
    input_size=784,  # Input size for MNIST/Fashion-MNIST (28x28 pixels flattened)
    hidden_layers=[128]*4,  # Two hidden layers with 128 and 64 neurons
    output_size=10,  # 10 output classes (digits 0-9 or fashion categories)
    learning_rate=0.001,  # Learning rate
    optimizer="nadam",  # Adam optimizer
    weight_init="xavier",  # Xavier weight initialization
    activation="tanh",  # ReLU activation function
    weight_decay=0.0005,  # L2 regularization strength
    loss_function="cross_entropy"  # Use cross-entropy loss
)

# Train the model and collect loss & accuracy history
cross_entropy_losses, cross_entropy_accuracies = cross_entropy_model.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=64)

# Train with Squared Error Loss 
squared_error_model = ModifiedNeuralNetwork(
    input_size=784,  
    hidden_layers=[128]*4,  
    output_size=10,  
    learning_rate=0.001,  
    optimizer="nadam",  
    weight_init="xavier",  
    activation="tanh",  
    weight_decay=0.0005,  
    loss_function="squared_error"  # Use squared error loss
)

# Train the model and collect loss & accuracy history
squared_error_losses, squared_error_accuracies = squared_error_model.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=64)

# Log Both Loss & Accuracy Comparisons 

# Save Loss Comparison Plot
loss_plot_path = "loss_comparison.png"  # File name for the loss plot
plt.figure(figsize=(10, 5))  # Set figure size
plt.plot(cross_entropy_losses, label="Cross-Entropy Loss", color="blue")  # Plot cross-entropy loss
plt.plot(squared_error_losses, label="Squared Error Loss", color="red", linestyle="dashed")  # Plot squared error loss
plt.xlabel("Epochs")  # Label for x-axis
plt.ylabel("Loss")  # Label for y-axis
plt.title("Loss Comparison: Cross-Entropy vs. Squared Error")  # Title for the plot
plt.legend()  # Show legend
plt.grid()  # Show grid lines
plt.savefig(loss_plot_path)  # Save the loss plot to a file
plt.close()  # Free memory after saving

# Save Accuracy Comparison Plot
accuracy_plot_path = "accuracy_comparison.png"  # File name for the accuracy plot
plt.figure(figsize=(10, 5))  # Set figure size
plt.plot(cross_entropy_accuracies, label="Cross-Entropy Accuracy", color="blue")  # Plot accuracy for cross-entropy
plt.plot(squared_error_accuracies, label="Squared Error Accuracy", color="red", linestyle="dashed")  # Plot accuracy for squared error
plt.xlabel("Epochs")  # Label for x-axis
plt.ylabel("Accuracy (%)")  # Label for y-axis
plt.title("Accuracy Comparison: Cross-Entropy vs. Squared Error")  # Title for the plot
plt.legend()  # Show legend
plt.grid()  # Show grid lines
plt.savefig(accuracy_plot_path)  # Save the accuracy plot to a file
plt.close()  # Free memory after saving

# Log both images with unique keys in WandB
wandb.log({
    "Loss Comparison Plot (CrossEntropy vs SquaredError)": wandb.Image(loss_plot_path),  # Log loss plot
    "Accuracy Comparison Plot (CrossEntropy vs SquaredError)": wandb.Image(accuracy_plot_path)  # Log accuracy plot
})

wandb.finish()  # End the WandB run
