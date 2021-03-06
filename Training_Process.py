"""
    Dec 27
    Minh T Nguyen
    CNN Training Pytorch:
    1/ Get batch from the training set
    2/ Pass batch to network
    3/ Calculate loss
    4/ Calculate the gradient of the loss function wrt (with respect to) the network's weights
    5/ Update the weight using gradients to reduce the loss
    6/ Repeat 1-5 until one epoch is completed
    7/ Replete 1-6 as many epoch as want ed to obtain the desired level of accuracy (Training loop)
    8/ Building a Confusion Matrix Display
    9/ Work with Tensorboard: Import + SummaryWriter()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # For updatding weights

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from resource.plotcm import plot_confusion_matrix

torch.set_printoptions(linewidth=120)  # Display options for output
torch.set_grad_enabled(True)

from torch.utils.tensorboard import SummaryWriter  # Work with tensorboard
from itertools import product

"""
    How to use Tensorboard?
    Terminal: ls .\runs\
              Check whatever directory created => cd to that then ls just to check
              tensorboard --logdir=runs (at the directory containing the .py file)
"""
#####################################################################3

# Calculate loss
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# Create network class: constructor and forward calculation
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) Input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) Hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)

        return t


# Import training set:
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# Training Process
# For Cartesian Product so we can run multiple option parallelly to maximize usage of TensorBoard
parameters = dict(
    lr = [0.01, 0.001],
    batch_size = [100,1000],
    shuffle = [True, False]
)
param_values = [v for v in parameters.values()]
for lr, batch_size, shuffle in product(*param_values): # * is a special way Python unpack a list into set of arguments
    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'

    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Update the weight using optimizer SGD or Adam
    optimizer = optim.Adam(network.parameters(), lr=lr)

    # Step 9: Add in Tensorboard #########
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)  # for visualization

    tb = SummaryWriter(comment=comment)
    tb.add_image("images", grid)
    tb.add_graph(network, images)

    for epoch in range(5):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch
            preds = network(images)  # Forward cal
            loss = F.cross_entropy(preds, labels)

            # Zero out the gradient held in grad attribute of the weight since pytorch will add latest-calculating gradient with the previous gradient
            optimizer.zero_grad()

            # Calculate the gradient
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += get_num_correct(preds, labels)

        ######### tb: add in scalar value: loss, correct, accuracy, and changing histogram of bias, weight, and gradient
        tb.add_scalar("loss", total_loss, epoch)  # add a number
        tb.add_scalar("Number Correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct / len(train_set), epoch)

        # Hard-code histogram for each layer
        # tb.add_histogram(
        #     "conv1.bias", network.conv1.bias, epoch
        # )  # Add a set of value for a histogram
        # tb.add_histogram("conv1.weight", network.conv1.weight, epoch)
        # tb.add_histogram("conv1.weight.grad", network.conv1.weight.grad, epoch)
        ######### tb

        # General Code histogram for all layers in the network
        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        # After this process the network do 60000/100 = 600 iterations => number of time the weights got updated
        print(
            "Epoch: ", epoch, " Total Correct: ", total_correct, " Total Loss: ", total_loss
        )
    tb.close()

print("Accuracy is: ", total_correct / len(train_set))

# ###################################################################################################################
# # Step 8: Building confusion matrix display
# # Function take in trained model and make prediction to all input features
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([])
#     for batch in loader: # Can't predict all 60000 images at once
#         images, labels = batch
#
#         preds = model(images)
#         all_preds = torch.cat((all_preds, preds), dim=0) # add the prediction of 1 batch into the list
#     return all_preds
#
# with torch.no_grad(): # compute without tracking gradient in prediction steps
#     prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
#     train_preds = get_all_preds(network, prediction_loader) # Should be [60000,10]
#
# preds_correct = get_num_correct(train_preds, train_set.targets)
#
# print("Total correct: ", preds_correct)
# print("Accuracy: ", preds_correct/len(train_set))
#
# # Building the confusion matrix by pairing elementwise training labels with prediction argmax
# stacked = torch.stack(
#     (
#         train_set.targets,
#         train_preds.argmax(dim=1)
#     ),
#     dim=1
# )
#
# print(stacked) # For debugging
#
# cmt = torch.zeros(10,10, dtype=torch.int32) # Building an empty confusion matrix
#
# for p in stacked:
#     label,prediction = p.tolist()
#     cmt[label,prediction] = cmt[label,prediction] + 1
#
# print("The confusion matrix is: \n", cmt)
#
# # Plotting with mpl
# cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
# print(type(cm))
# print(cm)
#
# names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle Boots')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, names)

# ###################################################################################################################
# # Step 9: Using Tensorboard - just for showing images on tensorboard only
# tb = SummaryWriter()
#
# network = Network()
# images, labels = next(iter(train_loader))
# grid = torchvision.utils.make_grid(images)
#
# tb.add_image('images', grid)
# tb.add_graph(network, images)
# tb.close()

###################################################################################################################
# Used when calculate 1 batch
# print("Before the optimizer is used, altho we update the weights, the loss is still the same:")
# print(loss.item()) # Loss cross entropy
# print(get_num_correct(preds, labels)) # return the percentage of correct
#
# optimizer.step() # update the weight, step in direction of the loss function to minimum
# preds = network(images) # Calculate the prediction again
# new_loss = F.cross_entropy(preds, labels)
# print(new_loss.item())
# print(get_num_correct(preds,labels)) # percentage of getting correct image 15/100, note this is just 1 batch, 100 samples form 60000 samples.
