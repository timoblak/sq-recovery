import cv2
import torch
import h5py
import os
import glob
import pickle
import random
from time import time, sleep
from tqdm import tqdm
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.utils import data
from classes import H5Dataset, IoUAccuracy, ChamferQuatLoss
from helpers import parse_csv, plot_render, plot_grad_flow, getBack, quat2mat, get_command
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate, randquat, multiply2, to_magnitude
from helpers import load_model
from models import GenericNetSQ, ResNetSQ


if __name__ == "__main__":
    
    device = "cuda:0"
    scanner_location = "../data/"   

    TRAINED_MODEL = "trained_models/model_e4_32_32_angle_2.pt"
    N = 50
    DEBUG = True
    f = open("results.txt", "a")

    net = ResNetSQ(outputs=4, pretrained=False).to(device)
    acc_full = IoUAccuracy(render_size=128, device=device, full=True)
    acc = IoUAccuracy(render_size=128, device=device, full=False)
    loss = ChamferQuatLoss(render_size=32, device=device)

    # Load model weights and put network in to EVAL mode! 
    epoch, net, _, _ = load_model(TRAINED_MODEL, net, None)
    net.eval()

    # Set random seed for official benchmarks
    random.seed(1234)
    
    accs = []
    #with torch.no_grad():
    for i in tqdm(range(N)):
        # Create a random instance 
        a = np.random.uniform(25, 75, (3,)).astype("float64")
        e = np.random.uniform(0.1, 1.0, (2,)).astype("float64")
        t = np.array([128, 128, 128], dtype="float64") + np.random.uniform(-40, 40, (3,))
        q = randquat()
        M1 = quat2mat(q)

        # Prepare parameters (non-normalized) and create image
        params = np.concatenate((a, e, t, M1.ravel()))
        command = get_command(scanner_location, "tmp1.bmp", params)
        os.system(command)

        # Read image and predict on it 
        img1 = cv2.imread("tmp1.bmp", 0)
        img_np = np.expand_dims(np.expand_dims(img1.astype(np.float32), 0), 0)
        img_np = img_np/255

        data = torch.from_numpy(img_np).to(device)
        pred_labels = net(data)
        
        
        # Create parameters to evaluate 
        block_params = torch.tensor(np.expand_dims(np.concatenate((a/255, e, t/255)), 0), dtype=torch.double, device=device)
        quat_true = torch.tensor(np.expand_dims(q, axis=0), dtype=torch.double, device=device)
        
        true_param = torch.cat((block_params, quat_true), dim=-1)
        
        print("---------- Example "+str(i)+" ----------", file=f)
        print("True params:", a, e, t, q, file=f)
        ap, ep, tp = torch.split(pred_labels[0][0], (3, 2, 3))
        ap = ap.detach().cpu().numpy()*255
        ep = ep.detach().cpu().numpy()
        tp = tp.detach().cpu().numpy()*255
        qp = pred_labels[1][0].detach().cpu().numpy()
        print("Pred params:", ap, ep, tp, qp, file=f)
        
        #print("--------------")

        M2 = quat2mat(qp)
        params = np.concatenate((a, e, t, M2.ravel()))
        command = get_command(scanner_location, "tmp2.bmp", params)
        os.system(command)

        params2 = np.concatenate((ap, ep, tp, M2.ravel()))
        command = get_command(scanner_location, "tmp3.bmp", params2)
        os.system(command)

        img2 = cv2.imread("tmp2.bmp", 0)
        img3 = cv2.imread("tmp3.bmp", 0)
        image = np.hstack([img1, img2, img3])

        diff = multiply2(quat_true, conjugate(pred_labels[1]))
        mag = to_magnitude(diff)
        
        accuracy_full = acc_full(true_param.double(), torch.cat(pred_labels, dim=1).double())
        accuracy = acc(true_param.double(), pred_labels[1].double())
    
        true_grad = torch.tensor([list(np.concatenate([a/255, e, t/255, q]))], device='cuda:0', dtype=torch.double)
        pred_grad = torch.tensor([list(qp)], device='cuda:0', requires_grad=True, dtype=torch.double)
        l = loss(true_grad, pred_grad)
        
        #print(pred_grad.grad)
        l.backward()
        
        loss_np = l.detach().cpu().item()
        grads_np = pred_grad.grad.detach().cpu().numpy()
        loss_np = l.detach().cpu().numpy()
        acc_np = accuracy.detach().cpu().numpy()
        acc_np_full = accuracy_full.detach().cpu().numpy()
        mag_np = mag.detach().cpu().numpy()
        
        mag_deg = np.rad2deg(mag_np)

        cv2.imwrite("examples/img_"+str(i)+".png", image)

        print("- Loss quat:", loss_np, file=f)
        print("- Accuracy quat:", acc_np*100, file=f)
        print("- Accuracy full:", acc_np_full*100, file=f)
        print("- Magnitude difference:", mag_deg[0], file=f)
        print("- Grads:", grads_np, file=f)

        accs.append([acc_np, acc_np_full, mag_np])
        
    accs = np.array(accs)
    a = np.array(accs[:,0])
    a_full = np.array(accs[:,1])
    m = np.array(accs[:,1])
    print("--Rot::")
    print("Mean: ", a.mean()) 
    print("Std: ", a.std())      
    print("--Full::")
    print("Mean: ", a_full.mean()) 
    print("Std: ", a_full.std())
    with open("accs_angle1.pkl", "wb") as handle: 
        pickle.dump([a,m], handle)
