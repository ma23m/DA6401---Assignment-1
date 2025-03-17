import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for better visualization of confusion matrix
from sklearn.metrics import confusion_matrix  # Import confusion matrix function
import wandb  # Import Weights & Biases for experiment tracking
from neural_network import NeuralNetwork  # Import the NeuralNetwork class
from dataset_loader import load_dataset  # Import dataset loading function

# Best Hyperparameters 
# -ac_tanh-hs_128-epc_10-hl_4-regu_0.0005-eta_0.001-optmz_nadam-batch_64-wght_xavier
best_config = {
    'hidden_layers': [128] * 4,  
    'learning_rate': 0.001,  
    'optimizer': 'nadam',  
    'weight_init': 'xavier',  
    'activation': 'tanh',  
    'weight_decay': 0.0005,  
    'epochs': 10,  
    'batch_size': 64  
}

# Define Sweep for Confusion Matrix 
sweep_config_conf_matrix = {
    'method': 'grid',  
    'name': 'Confusion Matrix Sweep Final 1234',  
    'parameters': {
        'model_name': {'values': ['best_model']}  
    }
}

# Create the sweep in WandB
sweep_id_conf_matrix = wandb.sweep(sweep=sweep_config_conf_matrix, project='DA6401_Assignment1_ma23m011')


def train_best_model_and_plot_conf_matrix():
    """ Train best model, log test accuracy, and plot confusion matrix in WandB. """

    # Initialize WandB experiment
    wandb.init(project="DA6401_Assignment1_ma23m011", name="Best Model - Conf Matrix 1234")

    # Load Dataset 
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("fashion_mnist")

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

    test_accuracies = []  

    for epoch in range(num_epochs):
        best_model.train(x_train, y_train, x_val, y_val, epochs=1, batch_size=batch_size)

        # Forward pass on test data to get predictions
        y_pred_probs = best_model.forward(x_test)[0][-1]  
        y_pred = np.argmax(y_pred_probs, axis=1)  
        y_true = np.argmax(y_test, axis=1)  

        # Compute test accuracy
        accuracy = np.sum(y_pred == y_true) / y_true.shape[0]  
        test_accuracies.append(accuracy * 100)  

        print(f"Epoch {epoch+1}: Test Accuracy = {accuracy*100:.2f}%")  

        # Log test accuracy per epoch to WandB
        wandb.log({"Test Accuracy": accuracy * 100, "Epoch": epoch + 1})

    # Save Test Accuracy Plot 
    accuracy_plot_path = "test_accuracy_plot.png"
    plt.figure(figsize=(8, 5))  
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', linestyle='-', color='b')  
    plt.xlabel("Epochs")  
    plt.ylabel("Test Accuracy (%)")  
    plt.title("Test Accuracy Over Epochs")  
    plt.grid()  
    plt.savefig(accuracy_plot_path)  # Save the plot
    plt.close()  

    # Log Test Accuracy Plot with unique key
    wandb.log({"Test Accuracy Plot": wandb.Image(accuracy_plot_path)})

    # Compute Confusion Matrix 
    cm = confusion_matrix(y_true, y_pred)  

    # Save Confusion Matrix Plot 
    conf_matrix_path = "confusion_matrix_plot.png"
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))  
    plt.xlabel("Predicted Label")  
    plt.ylabel("True Label")  
    plt.title(f"Confusion Matrix - Final Test Accuracy: {accuracy*100:.2f}%")  
    plt.savefig(conf_matrix_path)  # Save the plot
    plt.close()  

    # Log Confusion Matrix with unique key
    wandb.log({"Confusion Matrix": wandb.Image(conf_matrix_path)})


# Run the Sweep 
if __name__ == "__main__":
    wandb.agent(sweep_id_conf_matrix, function=train_best_model_and_plot_conf_matrix)
    wandb.finish()
