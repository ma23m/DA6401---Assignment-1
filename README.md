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

### Running the code:
After downloading all the py files please run the following commands to see the results of the respective questions: 

#### Command-Line Arguments

The following arguments can be used to configure the training process:

| Argument | Description |
|----------|-------------|
| --wandb_project | Project name used for tracking experiments in WandB. |
| --wandb_entity | WandB entity (`username`) for tracking experiments. |
| --dataset | Dataset choice (`mnist` or `fashion_mnist`). |
| --hidden_size | Number of `neurons` per hidden layer. |
| --num_layers | Number of `hidden layers` in the neural network. |
| --learning_rate | `Learning rate` for model training. |
| --optimizer | `Optimizer` choice for training. |
| --weight_init | `Weight initialization` method. |
| --activation | `Activation function` used in hidden layers. |
| --weight_decay | `Weight decay` (L2 regularization strength). |
| --epochs | Number of training `epochs`. |
| --batch_size | `Batch size` for training. |

#### Example
To train the model with custom arguments, please run:
```
python train.py --wandb_project "DA6401_Assignment1_ma23m011" --wandb_entity "ma23m011-iit-madras" --dataset "mnist" --epochs 10 --batch_size 32 --hidden_size 128 --num_layers 3 --learning_rate 0.001 --optimizer "adam" --weight_init "xavier" --activation "relu" --weight_decay 0.0005
```

#### 1. For question 1 run the following command 
```
python Q1_dataPlot.py
```

#### 2. For question 2 all the required details has been implemented in 
```
neural_network.py  file
```

#### 3. For question 3 all the required optimizers has been created in 
```
neural_network.py  file
```
under the method `update_weights`

#### 4. For question 4 all the required hyperparameter search has been executed from `train.py` file by the following command
```
python train.py --wandb_entity "ma23m011-iit-madras" --wandb_project "DA6401_Assignment1_ma23m011" --sweep
```

#### 5. For question 5
After training the model, the best validation accuracy is 89.1.

#### 6. For question 6 Parallel co-ordinates plot and a correlation summary has been shown in the respective section in the report

#### 7. For question 7 Please run the following command to generate Confusion Matrix and the respective Test Accuracy Plot over epochs
```
python Q7_ConfMatrix.py
```

#### 8.For question 8 Please run the following command to see the comparative analysis for the Squared Error Vs. Cross-Entropy Losses 
```
python Q8_CompareLosses.py
```

#### 10.For question 10 Please run the following command to explore the performance of best three configurations obatined from the `Fashion-MNIST` dataset over the `MNIST` dataset 
```
python Q10_mnistEvaluation.py
```

### Wandb Project Report: [Report Link](https://wandb.ai/ma23m011-iit-madras/DA6401_Assignment1_ma23m011/reports/Mousina-Barman-MA23M011-DA6401-Assignment-1--VmlldzoxMTQ5ODUxOA)






