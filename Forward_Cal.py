"""
    Forward Calculation
    Batch Processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train = True,
    download= True,
    transform=transforms.Compose([transforms.ToTensor()])
)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self,t):
        t = F.relu(self.conv1(t))
        # Max pool = pool out the max value at every location
        t = F.max_pool2d(t, kernel_size=2, stride=2) # Max pool 2d = filter has size of 2, but also the stride of 2

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu((self.fc1(t.reshape(-1, 12*4*4))))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

# Turn feature off to reduce memory consumption
torch.set_grad_enabled(False)

network = Network()
# image, label = next(iter(train_set))
#
# # Turn the image into single image batch size
# print(image.unsqueeze(0).shape)
#
# pred = network(image.unsqueeze(0))
# print(pred)
# print(pred.argmax(dim=1)) # Provide discrete value, if we want to output prob, use softmax
# print(label)

# Batch Processing
data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = 10
)

batch = next(iter(data_loader)) # iter make data into stream of node pair that you can iterate and grab adata
images, labels = batch
print(images.shape)
print(labels.shape)

pred = network(images)
print(pred)
print("The predictions are: ",pred.argmax(dim=1))
print("The labels are: ", labels)
print("Compare: ", pred.argmax(dim=1).eq(labels))
print("Total number of equal: ", pred.argmax(dim=1).eq(labels).sum())

# How to calculate the output of the CNN?
"""
    nxn input
    fxf fileter
    padding p and stride s
    output size O = [(n-f+2p)/s] + 1 => Calculate the height and width output of the cnn
    => This is how we calculate the maxpool 12*4*4
    
    Say we have input of 28*28
    conv1 (5x5) O = [(28-5)+0]/1 + 1 = 24
    maxpool2D (2x2) O = [(24-2) + 0]/2 + 1 = 12
    conv2 (5x5) O = [(12-5)+0]/1 + 1 = 8
    maxpool2D (2x2) O = [(8-2) + 0]/2 + 1 = 4 => 4x4  
"""
for name, layer in network.named_parameters():
    print(name, "\t\t", layer.shape)
