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
    11/ Clean up code with Run Manager Class: Avoid writing tb.add_scalar/add_histogram

    How to use Tensorboard?
    Terminal: ls .\runs\
              Check whatever directory created => cd to that then ls just to check
              tensorboard --logdir=runs (at the directory containing the .py file)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # For updatding weights
import torchvision
import torchvision.transforms as transforms
torch.set_printoptions(linewidth=120)  # Display options for output
torch.set_grad_enabled(True)
from torch.utils.tensorboard import SummaryWriter  # Work with tensorboard

# Packages for Training Loop Run Builder
from itertools import product
from collections import OrderedDict
from collections import namedtuple

import time
from IPython.display import display, clear_output
import pandas as pd
import json

###############################################################################
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

# Step 11: Run Manager Class to keep track of param
class RunManager():
    def __init__(self):
        # for each epoch
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        # for Run Param
        self.run_params = None
        self.run_count = 0 # run number
        self.run_data = []
        self.run_start_time = None # run duration

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network # Save network
        self.loader = loader # save data loader
        self.tb = SummaryWriter(comment=f' -{run}')

        images, labels = next(iter(self.loader)) # single batch of images
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar("loss", loss, epoch)  # add a number
        self.tb.add_scalar("Accuracy", accuracy / len(train_set), epoch)

        # General Code histogram for all layers in the network
        for name, weight in network.named_parameters():
            self.tb.add_histogram(name, weight, epoch)
            self.tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = run_duration

        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)

    def track_loss(self,loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
###############################################################################
# Import training set:
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

###############################################################################
# List of params that we want to try out
params = OrderedDict(
    lr = [0.01],
    batch_size = [1000, 2000],
    shuffle = [True, False]
)

m = RunManager()
# Instead of unpacking like above, we can just call the RunBuilder function
for run in RunBuilder.get_runs(params):
    comment = f' --{run}'
    network = Network()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)

    # Update the weight using optimizer SGD or Adam
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, train_loader)

    for epoch in range(5):
        m.begin_epoch()
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

            m.track_loss(loss)
            m.track_num_correct(preds,labels)

        m.end_epoch()
    m.end_run()
m.save("results")
