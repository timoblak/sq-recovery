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

        plot_render(self.xyz.cpu().numpy(), a[0].detach().cpu().numpy(), mode="in", figure=1)
        plot_render(self.xyz.cpu().numpy(), b[0].detach().cpu().numpy(), mode="in", figure=2)
        plt.show()
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

    def __call__(self, pred, true):
        a = self._ins_outs(pred)
        b = self._ins_outs(true)

        diff = a - b

        vis_a = a[0].detach().cpu().numpy()
        vis_b = b[0].detach().cpu().numpy()

        #plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2)
        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
#        plt.show()

        loss_val = torch.sqrt(torch.stack(list(map(torch.mean, torch.pow(diff, 2)))))
        if self.reduce:
            return loss_val.mean()
        return loss_val

if __name__ == "__main__":
    try:
        device = torch.device("cuda:0")
        req_grad = False
        closs_quat = ChamferLoss(26, device, reduce=True)
        closs_iso = ChamferLossIso(26, device, reduce=True)

        loss_mse = torch.nn.MSELoss(reduction="mean")


        #true = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0')
        #pred = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0', requires_grad=True)

        #true_mse = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0')
        #pred_mse = torch.tensor([[0.5630, 0.7245, 0.4229, 0.8618, 0.8556, 0.5738, 0.3802, 0.5493, 0, 0, 0, 1]], device='cuda:0', requires_grad=True)

        lr = 0.0001
        grads = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        vec = np.array([0.2630, 0.6245, 0.2229, 1, 1, 0.5738, 0.3802, 0.9093])
        while True:

            vec -= grads * lr
            true_iso = torch.tensor([[0.5630, 0.7245, 0.4229, 1, 1, 0.5338, 0.3802, 0.5593]], device='cuda:0')
            pred_iso = torch.tensor([vec], device='cuda:0', requires_grad=True)

            true_iso = torch.tensor([[0.5630, 0.7245, 0.4229, 1.0000, 1.0000, 0.5338, 0.3802, 0.5593]],
                   device='cuda:0')
            pred_iso = torch.tensor([[0.5485, 0.7017, 0.4249, 1.2284, 0.9437, 0.5336, 0.3814, 0.5595]],
                   device='cuda:0', dtype=torch.float64, requires_grad=True)

            print(true_iso)
            print(pred_iso)
            l = closs_iso(true_iso, pred_iso)
            l.backward()
            print("-----ISOMETRY LOSS------")
            print(l)

            grads = pred_iso.grad.cpu().numpy()[0]
            print(grads)

            print("----------------------------------------------------------------")
            print("----------------------------------------------------------------")
            del l
            sleep(0.5)

        """
        l = closs_quat(true, pred)
        l.backward()
        print("-----QUATERNION LOSS------")
        print(l)
        print(pred.grad)
        print("-----------")
        """

        """
        l = loss_mse(true_mse, pred_mse)
        l.backward()
        print("-----MSE LOSS------")
        print(l)
        print(pred_mse.grad)
        print("-----------")
        """

        plt.show()
    except KeyboardInterrupt:
        print("INTERRUPT")
        plt.close("all")
