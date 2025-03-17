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

#Initialize empty list to store sample image and their labels
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