"""
    Minh Nguyen
    Implement Sequential Model and Batch Normalization

    Batch Norm:
        mean and std are calculated with respect to the batch at normalization is aapplied.
        Two learnable param are use allow the data to be scaled (multiplication operration) and shifted (additiona operation)
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
# Import training set:
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]), # composition of transformation (turn image into a tensor)
)

# a loader here is to used for batch_normalization
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000,
                                           num_workers=1)
num_of_pixels = len(train_set) * 28 * 28

total_sum = 0
for batch in train_loader: total_sum += batch[
    0].sum()  # total values of all pixels in an image
mean = total_sum / num_of_pixels

sum_of_squared_error = 0
for batch in train_loader: sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
std = torch.sqrt(sum_of_squared_error / num_of_pixels)

# Normalize the training set with means and std
train_set_normal = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # composition of transformation (turn image into a tensor)
)

# Create a dictionary of regular training set and normalize trainining set
trainsets = {
    'not_normal': train_set,
    'normal': train_set_normal
}

###############################################################################
# Sequential model without batch normalization
torch.manual_seed(50)
network1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
)

# Sequential model with batch normalization
torch.manual_seed(50)
network2 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.BatchNorm2d(6) # out_features
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.BatchNorm1d(120) # out_features
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
)

networks = {
    'no_batch_norm': network1
    ,'batch_norm': network2
}

###############################################################################
params = OrderedDict(
    lr = [.01]
    , batch_size = [1000]
    , num_workers = [1]
    , device = ['cpu']
    , trainset = ['normal']
    , network = list(networks.keys())
)

# Calculate loss
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class RunBuilder():
    @staticmethod # Since get_rus is static we can call it using the class itself
    def get_runs(params):
        # Create a tuple subclass call Run taking in Class name and field name
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()): # create a Cartesian product and return an ordered pairs that define our run
            runs.append(Run(*v)) # use * to tell the constructor to accept the tuple values as argument rather than the tuple itself
        return runs

# Run Manager Class to keep track of param
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

    @torch.no_grad() # flag for local function
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
# Training Process
m = RunManager()

for run in RunBuilder.get_runs(params):
    comment = f' --{run}'

    device = torch.device(run.device)
    network = networks[run.network].to(device)

    loader = torch.utils.data.DataLoader(
        trainsets[run.trainset],
        batch_size = run.batch_size,
        num_workers = run.num_workers
    )

    # Update the weight using optimizer SGD or Adam
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, loader)

    for epoch in range(1):
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

