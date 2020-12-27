"""
    Learning Pytorch
    Dec 25
    Curriculum: Tensor, MNIST Fashion
"""

"""
    Parallel Computing: GPU
        Breaking large computation into small computation then merge them in the end
        Parallel: CNN: The computation of convolution does not relate to each other, meaning that the result of one computation does not relate to the computation of other parallel computation
        Nvidia = hardware to do parallel computation
        CUDA = software layer interact with Nvidia
"""

import torch
import numpy as np

# 1/ Using Cuda
# t = torch.Tensor([1,2,3]) # on cpu
# print(t)
#
# cuda_t = t.cuda()
# print(cuda_t)
#################################################################

# 2/ Reshaping
# Rank = number of dimention in a matrix
#a = [1,2,3,4,5,6]
# t = torch.tensor(a)
# t1 = t.reshape(3,2)
# print(t1)
#################################################################

# 3/ The shape of the cnn input has the len of 4
# [B,C,H,W] = batch_size, Color(3 = RGB, 1 = grey scale, height, width) of the image
# Output features maps are the product of the cnn taking in an input color channel corresponding to filters
#################################################################


# 4/ Torch Tensor Class
# t = torch.Tensor() # Create a tensor class
# print(t.dtype)
# print(t.device) #
# print(t.layout) # Stride: How tensor data layout in memory, default
#
# # Note that computation between tensor data type has to be the same dtype and device (cuda or cpu)
# t1 = torch.tensor([1,2,3,3])
# t2 = torch.tensor([1,1,1,1])
# print(t1.dtype)
# t4 = torch.tensor([1.0, 1.0,1.0]) # tensor will reserve the data type of the numpy
# print(t4.dtype)
# t5 = torch.Tensor([1,2,3,4])
# print(f"Dtype of Tensor is: {t5.dtype}") # Tensor will convert any dtype into float32
# t3 = t1 + t2
# print(f"The type of t3 is {t3.dtype} and the result is {t3}")
#################################################################


# # 5/ Transform Pytorch Tensor
# data = np.array([1,2,3])
# t = torch.tensor(data)
# print(t)
# t2 = torch.as_tensor(data)
# print(t2)
# data[0] = 100
# print(t2) # as_tensor share the same memory with the data
#################################################################


# 6 Reshaping operation
# data = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
# print(data)
# t = torch.tensor(data, dtype = torch.float32)
# print(t)
# print(f"The Shape of the tensor is: {t.size()}") # Specify the shape in torch
# print(f"The Rank of the tensor is: {len(t.shape)}") # number of dimention, shape does not have () since we want to return the value of len() but allow shape to run normally without returning anything
# print(f"The shape in numpy is: {np.shape(t)}")
# print(f"The number of elements in the tensor is {t.numel()}")
#
# t1= t.reshape(1,2,8)
# print(t1.shape)
# # Squeezing = delete all dimension that has the len of 1
# t2 = t1.squeeze()
# print(t2.shape)
# t2 = t2.unsqueeze(dim = 0)
# print(t2.shape)

# # Flattening = make any shape tensor into an array
# def flatten(t):
#     t = t.reshape(1,-1) # -1 means tensor can have anyshape, pytorch will figure this number out
#     t = t.squeeze() # Delete the dimension with size of 1
#     return t
# data = np.array([[1,1,1],[2,2,2],[3,3,3]])
# t = torch.tensor(data)
# print(t)
# t1 = flatten(t)
# print(f"The flatten t1 is now: {t1}")

# Try to concatenate the tensor
# data1 = np.array([[1,1,1],[2,2,2]])
# data2 = np.array([[3,3,3],[4,4,4]])
#
# t1 = torch.tensor(data1)
# t2 = torch.tensor(data2)
#
# print(f"{t1} and {t2}")
#
# t3 = torch.cat((t1,t2), dim = 0)
# print(f"{t3} has the shape of {t3.shape}")
#
# t4 = torch.cat((t1,t2), dim = 1)
# print(f"{t4} has the shape of {t4.shape}")
#################################################################


# # 7/ CNN Flatten Operation Visualize + stacking
# t1 = torch.tensor([[1,1,1],[2,2,2]])
# t2 = torch.tensor([[3,3,3],[4,4,4]])
# t3 = torch.tensor([[5,5,5],[6,6,6]])
#
# t4 = torch.stack((t1,t2,t3), dim=0)
# print(t4)
# print(t4.shape)
#
# t4 = t4.flatten(start_dim=1)
# print(t4) # Flatten each image
#
# # Flatten by using reshapping
# t4 = t4.reshape(3,6)
# print(t4)
# print(t4.shape)
#################################################################

# 8/ ArgMax and reduction operation
# # Reduction operation on a tensor = the operation that resduce the number of elements contained within the tensor
# # Reduction operation allow us to do arithmetic between elements within the same tensor
# t = torch.tensor([[1,1,1],[2,2,2]])
# t_sum = t.sum()
# print(t_sum < 9)
#
# # do reduction operation axis-wise
# t_e_sum = t.sum(dim = 0) # dim 0 return the array  [1,1,1] and [2,2,2], then it operate elements wise between the first dimension
# print(t_e_sum)
# t_sum2 = t.sum(dim = 1) # return the 2 array but do element wise operation between the elements within the array itself
# print(t_sum2)
#
# # Understanding: in dim = 0
# t1 = t[0]
# t2 = t[1]
# t3 = t1 + t2
# print(t3)
#
# # in dim = 1
# t1 = t[0].sum()
# t2 = t[1].sum()
# t3 = torch.tensor([t1,t2])
# print(t3)

# ARGMAX returns the index of the largest elements in the tensor#################################################################
t = torch.tensor([[1,2,3],[100,2,2]])
print(t.flatten())
print(f"The largest number is {t.max()} at position {t.argmax()}")
#################################################################
