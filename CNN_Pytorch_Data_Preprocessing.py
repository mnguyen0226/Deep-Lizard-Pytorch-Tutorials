"""
    Building a CNN in Pytorch
    1/ Data Preprocessing: ETL = Extract, transform, load data
    2/ Build the model
    3/ Train the model
    4/ Analyze the model's results
"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms # allow for data manipulation
import matplotlib.pyplot as plt
import matplotlib
"""
    About Torchvision - computer vision task
        Dataset (MNIST,etc)
        Models (VGG16...)
        Transforms (Data manipulation)
        Utils
"""

# Set the pytorch to the output console
torch.set_printoptions(linewidth = 120)

# Load data
training_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train = True, # for training set 60000
    download= True, # download if not in the dir already
    transform = transforms.Compose([transforms.ToTensor()]) # transform the image into tensor objects
)
# print(np.shape(training_set[0][0])) # Image of 28x28 size - feature
# print(np.size(training_set[0][1])) # labels

# Wrap data around the DataLoader objects for batch_size, shuffle,...
training_loader = torch.utils.data.DataLoader(training_set, batch_size = 10)

print(len(training_set))
print(training_set.train_labels) # to see the labels of each corresponding images
print(training_set.train_labels.bincount()) # Count the number of data in each clothing type => uniforms and balance

# Note: Jeremy Howard: The training dataset should be uniform, if it is not then replicate the lesser one until uniform
# # Access the data from training set
# image, label = next(iter(training_set)) # iter = turn the training dataset into a stream of objects that we can iterate over
# print(image.shape)
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()
# print(label)  # first label as tensor

# # Since we have training loader with batch size of 10 we can access image the same
# images, labels = next(iter(training_loader))
# print(images.shape) # [10,1,28,28] = 10 image, grayscale, 28 height, 28 width
#
# grid = torchvision.utils.make_grid(images, nrow = 10)
# plt.figure(figsize=(28,28))
# plt.imshow(np.transpose(grid, (1,2,0))) # Transpose image to be 28x28x1 to show the image
# plt.show()
# print("Labels: ", labels)


