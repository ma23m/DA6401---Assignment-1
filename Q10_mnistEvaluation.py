import wandb  # Import Weights & Biases for experiment tracking
from keras.datasets import mnist  # Import MNIST dataset from Keras
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from dataset_loader import load_dataset  # Import dataset loading function
from ModifiedNN_ForQ8_Q10 import ModifiedNeuralNetwork  # Import modified neural network class

# Load and Preprocess MNIST 
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("mnist")

# Train and Evaluate Function 
def train_and_evaluate(config, name):
    """ Train the model with the given configuration and evaluate performance. """
    
    wandb.init(project="DA6401_Assignment1_ma23m011", name=name, reinit=True)  # Initialize WandB for logging

    # Initialize model with given configuration
    model = ModifiedNeuralNetwork(
        input_size=784,  
        hidden_layers=config['hidden_layers'],  
        output_size=10,  
        learning_rate=config['learning_rate'],  
        optimizer=config['optimizer'],  
        weight_init=config['weight_init'],  
        activation=config['activation'],  
        weight_decay=config['weight_decay']  
    )

    num_epochs = 10  
    loss_values = []  
    test_accuracies = []  

    for epoch in range(num_epochs):  
        # Train for 1 epoch
        train_loss, _ = model.train(x_train, y_train, x_val, y_val, epochs=1, batch_size=config['batch_size'])
        loss_values.append(train_loss[-1])  

        # Forward pass for test accuracy
        y_pred_probs = model.forward(x_test)[0][-1]  
        y_pred = np.argmax(y_pred_probs, axis=1)  
        y_true = np.argmax(y_test, axis=1)  
        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]  
        test_accuracies.append(accuracy * 100)  

        # Log to WandB with unique epoch-wise keys
        wandb.log({f"{name} Epoch": epoch + 1, f"{name} Test Accuracy": accuracy * 100, f"{name} Loss": train_loss[-1]})

    # Save Loss Plot (Avoid Overwriting) 
    loss_plot_path = f"loss_plot_{name}.png"  
    plt.figure(figsize=(6, 4))  
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-', color='b')  
    plt.xlabel("Epochs")  
    plt.ylabel("Loss")  
    plt.title(f"{name} Loss Over Epochs")  
    plt.grid()  
    plt.savefig(loss_plot_path)  
    wandb.log({f"{name} Loss Plot": wandb.Image(loss_plot_path)})  
    plt.close()  

    # Save Accuracy Plot (Avoid Overwriting) 
    acc_plot_path = f"accuracy_plot_{name}.png"  
    plt.figure(figsize=(6, 4))  
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', linestyle='-', color='g')  
    plt.xlabel("Epochs")  
    plt.ylabel("Test Accuracy (%)")  
    plt.title(f"{name} Test Accuracy Over Epochs")  
    plt.grid()  
    plt.savefig(acc_plot_path)  
    wandb.log({f"{name} Accuracy Plot": wandb.Image(acc_plot_path)})  
    plt.close()  

    print(f"{name} - Final Test Accuracy: {test_accuracies[-1]:.2f}%")  
    return test_accuracies[-1]  

# Run 3 Best Configurations 
#-ac_tanh-hs_128-epc_10-hl_4-regu_0.0005-eta_0.001-optmz_nadam-batch_64-wght_xavier
#-ac_relu-hs_128-epc_10-hl_3-regu_0-eta_0.001-optmz_adam-batch_32-wght_xavier
#-ac_relu-hs_128-epc_10-hl_3-regu_0-eta_0.001-optmz_adam-batch_64-wght_xavier
configs = [
    {"hidden_layers": [128]*4, "activation": "tanh", "optimizer": "nadam", "learning_rate": 0.001, "batch_size": 64, "weight_decay": 0.0005, "weight_init": "xavier"},
    {"hidden_layers": [128]*3, "activation": "relu", "optimizer": "adam", "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.000, "weight_init": "xavier"},
    {"hidden_layers": [128]*3, "activation": "relu", "optimizer": "adam", "learning_rate": 0.001, "batch_size": 64, "weight_decay": 0.000, "weight_init": "xavier"}
]

results = []  
for i, config in enumerate(configs):  
    config_name = f"Config_{i+1}_Final"  
    accuracy = train_and_evaluate(config, config_name)  
    results.append((config_name, accuracy))  

# Print Final Results 
print("\nFinal Results on MNIST:")  
for config_name, acc in results:  
    print(f"{config_name}: {acc:.2f}%")  

wandb.finish()  # End WandB logging
