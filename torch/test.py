import cv2
import torch
import h5py
import glob
from time import time
from tqdm import tqdm
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.utils import data
from helpers import plot_render, plot_grad_flow, getBack
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate


class QuaternionLoss:
    def __init__(self, reduce=True):
        self.batch_reduce = reduce
        self.eps = 1e-8

    def __call__(self, ypred, ytrue):
        tmp = torch.abs(torch.sum(ytrue * ypred, dim=1))

        tmp = 0.5 + torch.abs(tmp - 0.5)

        theta = 2.0 * torch.acos(torch.clamp(tmp, -1 + self.eps, 1 - self.eps))
        print(theta)
        if self.batch_reduce:
            return torch.mean(theta)
        return theta

if "__main__" == __name__: 
    loss_quat = QuaternionLoss()
    x = Variable(torch.tensor([[0, 0, 0, 1], [0, 0, 0.7071068, 0.7071068], [0, 0, 1, 0]], dtype=float), requires_grad=True)
    y = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=float)
    #x = Variable(torch.tensor([[0, 0, 0, 1]], dtype=float), requires_grad=True)
    #y = torch.tensor([[0, 0, 0, 1]], dtype=float)
    loss = loss_quat(x, y)
    loss.backward()

    print("-------end--------")
    print(loss)
    print(x.grad)

