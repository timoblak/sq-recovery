import cv2
import torch
import h5py
import os
import glob
import pickle
from time import time, sleep
from tqdm import tqdm
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.utils import data
from classes import H5Dataset, IoUAccuracy
from helpers import parse_csv, plot_render, plot_grad_flow, getBack, quat2mat, get_command
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate, randquat
from helpers import load_model
from models import GenericNetSQ, ResNetSQ


if __name__ == "__main__":
    
    device = "cuda:0"
    scanner_location = "../data/"   

    TRAINED_MODEL = "trained_models/model_e4_32_8_2.pt"
    N = 100
    DEBUG = True
    
    net = ResNetSQ(outputs=4, pretrained=False).to(device)
    acc = IoUAccuracy(render_size=64, device=device)
    
    # Load model weights and put network in to EVAL mode! 
    epoch, net, _, _ = load_model(TRAINED_MODEL, net, None)
    net.eval()

    # Set random seed for official benchmarks 
    random.seed(1234)
    
    accs = []
    for _ in tqdm(range(N)):
        # Create a random instance 
        a = np.random.randint(25, 75, (3,)).astype("float64")
        e = np.random.uniform(0.1, 1.0, (2,)).astype("float64")
        t = np.array([128, 128, 128], dtype="float64") + np.random.uniform(-40, 40, (3,))
        q = randquat()
        M1 = quat2mat(q)

        # Prepare parameters (non-normalized) and create image
        params = np.concatenate((a, e, t, M1.ravel()))
        command = get_command(scanner_location, "tmp1.bmp", params)
        os.system(command)

        # Read image and predict on it 
        img_np = cv2.imread("tmp1.bmp", 0).astype(np.float32)
        img_np = np.expand_dims(np.expand_dims(img_np, 0), 0)
        img_np = img_np/255

        data = torch.from_numpy(img_np).to(device)
        pred_labels = net(data)

        # Create parameters to evaluate 
        block_params = torch.tensor(np.expand_dims(np.concatenate((a/255, e, t/255)), 0), dtype=torch.double, device=device)
        quat_true = torch.tensor(np.expand_dims(q, axis=0), dtype=torch.double, device=device)
        
        true_param = torch.cat((block_params, quat_true), dim=-1)
        
        accuracy = acc(true_param.double(), pred_labels.double())
        acc_np = accuracy.detach().cpu().numpy()
        print(acc_np)
        
        accs.append(acc_np)
        
    accs = np.array(accs)
    print("Mean: ", accs.mean()) 
    print("Std: ", accs.std())      

    with open("accs_improved_1.pkl", "wb") as handle: 
        pickle.dump(accs, handle)