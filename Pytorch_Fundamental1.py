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

# 1/ Using Cuda
# t = torch.Tensor([1,2,3]) # on cpu
# print(t)
#
# cuda_t = t.cuda()
# print(cuda_t)

# 2/ Reshaping
#a = [1,2,3,4,5,6]
# t = torch.tensor(a)
# t1 = t.reshape(3,2)
# print(t1)

# 3/ The shape of the cnn input has the len of 4
# [B,C,H,W] = batch_size, Color(3 = RGB, 1 = grey scale, height, width) of the image
