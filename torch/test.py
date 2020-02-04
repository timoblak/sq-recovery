import cv2
import torch
import h5py
import glob
from time import time, sleep
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


class ChamferLossIso:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce

        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)

    @staticmethod
    def preprocess_sq(p):
        a, e, t = torch.split(p, (3, 2, 3))
        a = (a * 0.196) + 0.098
        e = torch.clamp(e, 0.1, 1)
        return torch.cat([a, e, t], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            perameters = self.preprocess_sq(p[i])
            a, e, t = torch.split(perameters, (3, 2, 3))

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (self.xyz[0] - t[0]) / a[0]
            y_translated = (self.xyz[1] - t[1]) / a[1]
            z_translated = (self.xyz[2] - t[2]) / a[2]

            A1 = torch.pow(x_translated, 2)
            B1 = torch.pow(y_translated, 2)
            C1 = torch.pow(z_translated, 2)

            A = torch.pow(A1, (1 / e[1]))
            B = torch.pow(B1, (1 / e[1]))
            C = torch.pow(C1, (1 / e[0]))

            D = A + B
            E = torch.pow(D, (e[1] / e[0]))
            inout = E + C

            # inout = torch.pow(inout, e[1])
            results.append(inout)

        return torch.stack(results)

    def __call__(self, pred, true):
        a = self._ins_outs(pred)
        b = self._ins_outs(true)

        #plot_render(self.xyz.cpu().numpy(), a[0].detach().cpu().numpy(), mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), b[0].detach().cpu().numpy(), mode="in", figure=2)
        #plt.show()
        return F.mse_loss(a, b)

        #diff = a - b


        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.show()

        loss_val = torch.sqrt(torch.stack(list(map(torch.mean, torch.pow(diff, 2)))))
        if self.reduce:
            return loss_val.mean()
        return loss_val

class ChamferLoss:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce

        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = (a * 0.196) + 0.098
        e = torch.clamp(e, 0.1, 1)
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            perameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(perameters, (3, 2, 3, 4))

            # Create a rotation matrix from quaternion
            rot = mat_from_quaternion(q)

            # Rotate translation vector using quaternion
            t = rotate(t, q)

            # Rotate coordinate system using rotation matrix
            m = torch.einsum('xij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (m[0] - t[0]) / a[0]
            y_translated = (m[1] - t[1]) / a[1]
            z_translated = (m[2] - t[2]) / a[2]

            A1 = torch.pow(x_translated, 2)
            B1 = torch.pow(y_translated, 2)
            C1 = torch.pow(z_translated, 2)

            A = torch.pow(A1, (1 / e[1]))
            B = torch.pow(B1, (1 / e[1]))
            C = torch.pow(C1, (1 / e[0]))

            D = A + B
            E = torch.pow(torch.abs(D), (e[1] / e[0]))
            inout = E + C
            inout = torch.pow(inout, e[1])
            results.append(inout)

        return torch.stack(results)

    def __call__(self, true, pred):
        a = self._ins_outs(pred)
        b = self._ins_outs(true)
        diff = a - b

        #vis_a = a[0].detach().cpu().numpy()
        #vis_b = b[0].detach().cpu().numpy()

        #plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2)
        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.show()
        #return F.mse_loss(b, a)
        loss_val = torch.sqrt(torch.stack(list(map(torch.mean, torch.pow(diff, 2)))))
        if self.reduce:
            return loss_val.mean()
        return loss_val

if __name__ == "__main__":
    try:
        device = torch.device("cuda:0")
        req_grad = False
        closs_quat = ChamferLoss(32, device, reduce=True)
        closs_iso = ChamferLossIso(26, device, reduce=True)

        loss_mse = torch.nn.MSELoss(reduction="mean")
        w = [1, 0.001, 1, 1]

        #true = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0')
        #pred = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0', requires_grad=True)

        #true_mse = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0')
        #pred_mse = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0', requires_grad=True)

        lr = 0.000001
        grads = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        vec = np.array([0.0500, 0.0500, 0.1128, 0.1500, 0.6181, 0.3611, 0.0100, 0.4949, 0.2348, 0.6032, 0.2051, 0.7341])
        iteration = 0
        while True:

            vec -= grads * lr
            true = torch.tensor([[0.1084, 0.2769, 0.1893, 0.3213, 0.7323, 0.6345, 0.5084, 0.4530, -0.6100, 0.5794, -0.5103, 0.1780]], device='cuda:0', dtype=torch.float64)
            pred = torch.tensor([vec], device='cuda:0', requires_grad=True)

            a, e, t, q = torch.split(pred, (3, 2, 3, 4), dim=-1)
            a_true, e_true, t_true, q_true = torch.split(true, (3, 2, 3, 4), dim=-1)

            print(true.dtype, pred.dtype)
            l_a = closs_quat(true, torch.cat((a, e_true, t_true, q_true), dim=-1)) * w[0]
            l_e = closs_quat(true, torch.cat((a_true, e, t_true, q_true), dim=-1)) * w[1]
            l_t = closs_quat(true, torch.cat((a_true, e_true, t, q_true), dim=-1)) * w[2]
            l_q = closs_quat(true, torch.cat((a_true, e_true, t_true, q), dim=-1)) * w[3]

            l = l_a + l_e + l_t + l_q
            l.backward()
            print("-----ISOMETRY LOSS------")
            print("Iteration " + str(iteration))
            print("LR " + str(lr))
            print(l)

            grads = pred.grad.cpu().numpy()[0]
            print(grads)

            print("----------------------------------------------------------------")
            print(true)
            print(pred)
            print("----------------------------------------------------------------")

            iteration+= 1
            if iteration % 550 == 0:
                print("Changinh LR to " + str(lr))
                lr *= 0.1

    except KeyboardInterrupt:
        print("INTERRUPT")
        plt.close("all")
