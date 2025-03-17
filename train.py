import argparse  # Import argparse for command-line argument parsing
import wandb  # Import Weights & Biases for experiment tracking
import numpy as np  # Import NumPy for numerical operations
from neural_network import NeuralNetwork  # Import the NeuralNetwork class
from dataset_loader import load_dataset  # Import dataset loading function
from types import SimpleNamespace  # Import SimpleNamespace for handling configuration

# Argument Parsing 
def parse_args():
    """ Parse command-line arguments for configuring the neural network training. """
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST or Fashion-MNIST.")
    
    # WandB arguments for experiment tracking
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name for WandB")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="WandB entity for tracking experiments")

    # Dataset and training settings
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset choice")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")

    # Loss function and optimizer settings
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer")

    # Learning rate and optimization hyperparameters
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for Momentum/NAG")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6, help="Epsilon for optimizers")

    # Regularization and weight initialization
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method")

    # Neural network architecture settings
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")

    # Flag to run hyperparameter sweep
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")

    return parser.parse_args()  # Return parsed arguments


# Training Function 
def train(args):
    """ Train the neural network using the given configuration. """
    args = parse_args()  # Parse command-line arguments

    # WandB Login 
    wandb.login(key='580e769ee2f34eafdded556ce52aaf31c265ad3b')  # Login to WandB using the provided API key

    # Initialize WandB experiment
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Load dataset (MNIST or Fashion-MNIST)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(args.dataset)

    # Define network architecture (hidden layers)
    hidden_layers = [args.hidden_size] * args.num_layers  # Create a list with 'num_layers' hidden layers

    # Initialize the neural network model with parsed arguments
    model = NeuralNetwork(
        input_size=784,  # Input size for MNIST/Fashion-MNIST (28x28 flattened)
        hidden_layers=[args.hidden_size] * args.num_layers,
        output_size=10,  # 10 output classes (digits 0-9 or Fashion items)
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_init=args.weight_init,
        activation=args.activation,
        weight_decay=args.weight_decay
    )

    # Train the model
    model.train(x_train, y_train, x_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    # Compute test accuracy after training
    y_pred_probs = model.forward(x_test)[0][-1]  # Get final predictions
    y_pred = np.argmax(y_pred_probs, axis=1)  # Get predicted class labels
    y_true = np.argmax(y_test, axis=1)  # Get true labels
    accuracy = np.sum(y_pred == y_true) / y_true.shape[0]  # Compute accuracy

    print(f"Test Accuracy: {accuracy*100:.2f}%")  # Print final test accuracy
    wandb.log({"Test Accuracy": accuracy * 100})  # Log test accuracy in WandB


# Hyperparameter Sweep Configuration 
sweep_config = {
    'method': 'bayes',  # Bayesian optimization for hyperparameter search
    'name': 'sweep cross entropy new Final',
    'metric': {'name': 'Val_accuracy', 'goal': 'maximize'},  # Optimize for validation accuracy
    'parameters': {
        'epochs': {'values': [5, 10]},  # Number of training epochs
        'hidden_layers': {'values': [3, 4, 5]},  # Number of hidden layers
        'hidden_size': {'values': [32, 64, 128]},  # Neurons per hidden layer
        'weight_decay': {'values': [0, 0.0005, 0.5]},  # L2 regularization
        'learning_rate': {'values': [1e-3, 1e-4]},  # Learning rate options
        'optimizer': {'values': ['rmsprop', 'nadam', 'adam']},  # Optimizer selection
        'batch_size': {'values': [16, 32, 64]},  # Mini-batch sizes
        'weight_init': {'values': ['xavier', 'random']},  # Weight initialization methods
        'activation': {'values': ['relu', 'tanh', 'sigmoid']}  # Activation function choices
    }
}


# Sweep Function 
def sweep_train():
    """ Perform training using hyperparameter sweep. """
    with wandb.init() as run:
        # Set experiment name dynamically using hyperparameters
        wandb.run.name = f"-ac_{wandb.config.activation}-hs_{wandb.config.hidden_size}-epc_{wandb.config.epochs}-hl_{wandb.config.hidden_layers}-regu_{wandb.config.weight_decay}-eta_{wandb.config.learning_rate}-optmz_{wandb.config.optimizer}-batch_{wandb.config.batch_size}-wght_{wandb.config.weight_init}"
        
        # Load dataset
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("fashion_mnist")

        # Initialize neural network with selected hyperparameters
        nn = NeuralNetwork(
            input_size=784,
            hidden_layers=[wandb.config.hidden_size] * wandb.config.hidden_layers,
            output_size=10,
            learning_rate=wandb.config.learning_rate,
            optimizer=wandb.config.optimizer,
            weight_init=wandb.config.weight_init,
            activation=wandb.config.activation,
            weight_decay=wandb.config.weight_decay
        )

        # Train the neural network
        nn.train(x_train, y_train, x_val, y_val, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size)


# Main Execution 
if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    if args.sweep:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)  # Initialize WandB sweep
        wandb.agent(sweep_id, function=sweep_train, count=100)  # Run the sweep function
    else:
        train(args)  # Run normal training

    wandb.finish()  # End WandB run
