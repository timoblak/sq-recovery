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
from helpers import plot_render, plot_grad_flow, getBack, quat2mat, get_command
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate
from resnet import resnet18
from helpers import load_model


class IoUAccuracy:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device

        step = 2 / render_size
        range = torch.tensor(np.arange(-1, 1 + step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz.requires_grad = False


    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            a, e, t, q = torch.split(p[i], (3, 2, 3, 4))

            # Create a rotation matrix from quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            coordinate_system = torch.einsum('ij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = coordinate_system[0] / (a[0] * 2)
            y_translated = coordinate_system[1] / (a[1] * 2)
            z_translated = coordinate_system[2] / (a[2] * 2)

            # Calculate powers of 2 to get rid of negative values)
            A1 = torch.pow(x_translated, 2)
            B1 = torch.pow(y_translated, 2)
            C1 = torch.pow(z_translated, 2)

            # Then calculate root
            A = torch.pow(A1, (1 / e[1]))
            B = torch.pow(B1, (1 / e[1]))
            C = torch.pow(C1, (1 / e[0]))

            E = torch.pow(A + B, (e[1] / e[0]))
            inout = E + C
            
            inout = torch.pow(inout, e[0])
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred_quat):
        pred_quat = F.normalize(pred_quat, dim=-1)
        a = self._ins_outs(true)
        b = self._ins_outs(torch.cat((true[:, :8], pred_quat), dim=-1))
        
        a_bin = (a <= 1)
        b_bin = (b <= 1)
        
        intersection = a_bin & b_bin
        union = a_bin | b_bin
        
        return torch.sum(intersection).double()/torch.sum(union).double() 


def randquat():
    u = np.random.uniform(0, 1, (3,))
    q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                  np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                  np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                  np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
    return q

if __name__ == "__main__":
    
    device = "cuda:0"
    abs_errs = []
    net = resnet18(num_classes=4).to(device)
    acc = IoUAccuracy(render_size=64, device=device)
    epoch , net, _, _ = load_model("models/model_tanh_3x1d_dropout.pt", net, None)
    #np.random.seed(1234)
    #print(epoch)
    scanner_location = "../data/"

    N = 5000
    debug = True
    accs = []
    for _ in tqdm(range(N)):
        a = np.random.randint(25, 75, (3,)).astype("float64")
        e = np.random.uniform(0.1, 1.0, (2,)).astype("float64")
        t = np.array([128, 128, 128], dtype="float64") + np.random.uniform(-40, 40, (3,))
        q = randquat()
        
        M1 = quat2mat(q)
        params = np.concatenate((a, e, t, M1.ravel()))
        command = get_command(scanner_location, "tmp1.bmp", params)
        os.system(command)

        params_true = np.concatenate([params, q])

        img_np = cv2.imread("tmp1.bmp", 0).astype(np.float32) #/255
        img_np = np.expand_dims(np.expand_dims(img_np, 0), 0)
        img = torch.from_numpy(img_np).to(device)
        #print(img.shape)
        quat_pred = net(img)
        
        # M = quat2mat(quat_pred[0].detach().cpu().numpy())

        block_params = torch.tensor(np.expand_dims(np.concatenate((a/255., e, t/255.)), 0), dtype=torch.double, device=device)
        quat_true = torch.tensor(np.expand_dims(q, axis=0), dtype=torch.double, device=device)
        
        true_param = torch.cat((block_params, quat_true), dim=-1)
        pred_param = quat_pred
        
        
        print(true_param)
        print(pred_param)
        
        accuracy = acc(true_param.double(), pred_param.double())
        acc_np = accuracy.detach().cpu().numpy()
        print(acc_np)
        accs.append(acc_np)
        
    accs = np.array(accs)
    print("Mean: ", accs.mean()) 
    print("Std: ", accs.std())      

    with open("accs_3x1d_dropout.pkl", "wb") as handle: 
        pickle.dump(accs, handle)
