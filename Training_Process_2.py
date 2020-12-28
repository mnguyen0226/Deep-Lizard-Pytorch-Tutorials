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
    10/ Build a Training Loop Run Builder Class
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

# Packages for Training Loop Run Builder
from collections import OrderedDict
from collections import namedtuple

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

# Step 10: Create a Run Builder Class: contain params that set the run
class RunBuilder():
    @staticmethod # Since get_rus is static we can call it using the class itself
    def get_runs(params):
        # Create a tuple subclass call Run taking in Class name and field name
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()): # create a Cartesian product and return an ordered pairs that define our run
            runs.append(Run(*v)) # use * to tell the constructor to accept the tuple values as argument rather than the tuple itself
        return runs

# List of params that we want to try out
params = OrderedDict(
    lr = [0.01, 0.001],
    batch_size = [1000, 10000]
)

runs = RunBuilder.get_runs(params)

# Import training set:
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# Training Process
# # For Cartesian Product so we can run multiple option parallelly to maximize usage of TensorBoard
# parameters = dict(
#     lr = [0.01, 0.001],
#     batch_size = [100,1000],
#     shuffle = [True, False]
# )
# param_values = [v for v in parameters.values()]
# for lr, batch_size, shuffle in product(*param_values): # * is a special way Python unpack a list into set of arguments
#    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'


# Instead of unpacking like above, we can jsut call the RunBuilder function
for run in RunBuilder.get_runs(params):
    comment = f' --{run}'

    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=True)

    # Update the weight using optimizer SGD or Adam
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

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

            total_loss += loss.item() * run.batch_size
            total_correct += get_num_correct(preds, labels)

        ######### tb: add in scalar value: loss, correct, accuracy, and changing histogram of bias, weight, and gradient
        tb.add_scalar("loss", total_loss, epoch)  # add a number
        tb.add_scalar("Number Correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct / len(train_set), epoch)

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

