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
from torchsummary import summary
from torch.utils import data
from helpers import plot_render, plot_grad_flow, plot_points, norm_img, randquat, slerp
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate, multiply, to_magnitude, to_euler_angle, to_axis_angle


#torch.set_printoptions(sci_mode=True)
#torch.set_printoptions(profile="full", precision=10)

class H5Dataset(data.Dataset):
    def __init__(self, dataset_location, labels, train_split, dataset_file='dataset.h5'):
        self.labels = labels
        self.dataset_location = dataset_location
        self.ext = ".bmp"
        self.h5_dataset_file = dataset_file
        self.h5_filepath = self.dataset_location + self.h5_dataset_file

        self.dataset = None
        self.handle = None
        self.mode = 0  # 0 - train, 1 - validate

        self.n_train = int(train_split * len(labels))
        self.n_val = len(labels) - self.n_train

        self.build_dataset()

    def __len__(self):
        if self.mode == 0:
            return self.n_train
        return self.n_val

    def set_mode(self, mode):
        self.mode = mode

    def load_dataset(self):
        print("Opening dataset")
        self.dataset = h5py.File(self.h5_filepath, 'r')["sq"]

    def close(self):
        self.handle.close()

    def build_dataset(self):
        if glob.glob(self.h5_filepath):
            print("Using existing dataset" + str(self.dataset_location))
            return

        print("Building a new dataset " + str(self.dataset_location))
        file_list = sorted(glob.glob1(self.dataset_location, "*" + self.ext))

        with h5py.File(self.h5_filepath, 'w') as handle:
            ds = handle.create_dataset("sq", (len(file_list), 1, 256, 256), dtype='f')
            for i, img_name in enumerate(tqdm(file_list)):
                ds[i] = self.load_image(img_name)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Load data and get label
        #t_start = time()

        if self.mode == 1:
            index += self.n_train
        #print("Fetching index", index)
        with h5py.File(self.h5_filepath, 'r') as db:
            X = db["sq"][index] / 255

        #print("Dataset_access: ", time() - t_start)
        y = self.load_label(index)
        return torch.from_numpy(X), torch.from_numpy(y)

    def load_image(self, img_name):
        # Load image and reshape to (channels, height, width)
        img = cv2.imread(self.dataset_location + img_name, 0).astype(np.float)
        img = np.expand_dims(img, -1)
        img = np.transpose(img, (2, 0, 1))
        # Do preprocessing
        return img

    def load_label(self, ID):
        # Select labels and preprocess
        # ### First 8 for isometric model ####
        return self.labels[ID][:12]


class QuaternionLoss:
    def __init__(self, reduce=True):
        self.batch_reduce = reduce
        self.eps = 1e-8

    def __call__(self, ypred, ytrue):
        theta = torch.tensor(1) - 2 * torch.abs(
            torch.tensor(0.5) - torch.pow(torch.einsum('ji,ji->j', ytrue, ypred), 2))
        if self.batch_reduce:
            return torch.mean(theta)
        return theta


class ChamferLoss:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device

        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz[self.xyz == 0] += 1e-4
        self.xyz.requires_grad = False

    @staticmethod
    def preprocess_sq(p):
        #a, t, q = torch.split(p, (3, 3, 4))
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0, max=1)
        #return torch.cat([a, t, q], dim=-1)
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(parameters, (3, 2, 3, 4))

            # To transform info general space either:
            # - contruct 4x4 homogenous transformation matrix from q and t.
            #   then inverse and apply to xyz space
            # - take conjugate of q (or transpose of rotation matrix) and then
            #   first rotate translation vector, then rotate and translate the space <- impemented

            # Create a rotation matrix from conjugated quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            t = torch.matmul(rot, t)

            coordinate_system = torch.einsum('ij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (coordinate_system[0] - t[0]) / a[0]
            y_translated = (coordinate_system[1] - t[1]) / a[1]
            z_translated = (coordinate_system[2] - t[2]) / a[2]

            # Calculate powers of 2 to get rid of negative values)
            A1 = torch.pow(x_translated, 2)
            B1 = torch.pow(y_translated, 2)
            C1 = torch.pow(z_translated, 2)
            A1[A1 == 0] += 1e-4
            B1[B1 == 0] += 1e-4
            C1[C1 == 0] += 1e-4
            # Then calculate root
            A = torch.pow(A1, (1 / e[1]))
            B = torch.pow(B1, (1 / e[1]))
            C = torch.pow(C1, (1 / e[0]))

            E = torch.pow(A + B, (e[1] / e[0]))
            inout = E + C

            inout = torch.pow(inout, e[0])

            inout = torch.sigmoid(5 * (1 - inout))
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred):
        #print("-------------------------------------------")
        #pred = F.normalize(pred, dim=-1)
        a = self._ins_outs(true)

        b = self._ins_outs(pred)

        losses = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            l = torch.mean(torch.pow(ai - bi, 2)) * 100
            losses.append(l)
        ls = torch.mean(torch.stack(losses, dim=-1))

        return ls


general = True

class ChamferQuatLoss:
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
        self.xyz[self.xyz == 0] += 1e-4
        self.xyz.requires_grad = False

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            #parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(p[i], (3, 2, 3, 4))

            # Create a rotation matrix from quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            #t = torch.matmul(rot, t)
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
            #inout = torch.sqrt(inout)
            #inout = torch.sigmoid(inout*2)
            inout = torch.sigmoid(5 * (1 - inout))
            #inout = A1 + B1 + C1
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred_quat):
        a = self._ins_outs(true)
        b = self._ins_outs(torch.cat((true[:, :8], pred_quat), dim=-1))

        #plot_render(self.xyz.detach().cpu().numpy(), b[0].detach().cpu().numpy(), "in_inv", 1, (-1, 1))
        #plot_render(self.xyz.detach().cpu().numpy(), a[0].detach().cpu().numpy(), "in_inv", 2, (-1, 1))
        #plt.show()

        losses = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            l = torch.mean(torch.pow(ai - bi, 2)) * 100
            losses.append(l)
        ls = torch.mean(torch.stack(losses, dim=-1))

        #if ls < 0.001:
        #    plot_render(self.xyz.detach().cpu().numpy(), b[0].detach().cpu().numpy(), "in_inv", 1, (-1, 1))
        #    plot_render(self.xyz.detach().cpu().numpy(), a[0].detach().cpu().numpy(), "in_inv", 2, (-1, 1))
        #    plt.show()
        return ls


class RotLoss:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 0.7
        self.reduce = reduce

        step = 2 / render_size
        range = torch.tensor(np.arange(-1, 1 + step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.device = device

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = a * 2
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, parameters):
        parameters = self.preprocess_sq(parameters)
        a, e, t, q = torch.split(parameters, (3, 2, 3, 4))

        # #### Calculate inside-outside equation ####
        # First the inner parts of equation (only divide with axis sizes)
        m = self.xyz
        x_translated = m[0] / a[0]
        y_translated = m[1] / a[1]
        z_translated = m[2] / a[2]

        A1 = torch.pow(x_translated, 2)
        B1 = torch.pow(y_translated, 2)
        C1 = torch.pow(z_translated, 2)

        A = torch.pow(A1, (1 / e[1]))
        B = torch.pow(B1, (1 / e[1]))
        C = torch.pow(C1, (1 / e[0]))

        E = torch.pow(A + B, (e[1] / e[0]))

        inout = E + C
        inout = torch.pow(inout, e[0])
        return inout

    def __call__(self, true, pred_quat):
        #print("-------------------------------------------")
        true = true.double()
        pred = pred_quat.double()
        global general
        batch_errs = []
        for i in range(true.shape[0]):
            # Gererate the inout space
            inout = self._ins_outs(true[i])

            # determine the border around surface, that points are sampled from
            #t = time()
            for border in np.arange(0, 1, 0.01):
                indxs = torch.where((inout < (1 + border)) & (inout > (1 - border)))
                if indxs[0].shape[0] > 30:
                    break
            #print("Time searching:", time()-t)
            #border = 0.1 + self.eps * (1 - torch.min(true[i][3], true[i][4]))
            #indxs = torch.where((inout < (1 + border)) & (inout > (1 - border)))

            # plot_render(self.xyz.cpu().numpy(), inout.detach().cpu().numpy(), mode="shell", figure=1, lims=(-1, 1), eps=border.cpu().numpy())
            # plt.show()

            xs = self.xyz[0, indxs[0], indxs[1], indxs[2]]
            ys = self.xyz[1, indxs[0], indxs[1], indxs[2]]
            zs = self.xyz[2, indxs[0], indxs[1], indxs[2]]
            stacked = torch.stack([xs, ys, zs])

            points = stacked.permute(1, 0)
            #selection = min(80, points.shape[0])
            #if selection == 0:
            #    print(border)
            #    print(true[i])
            #    print(points.shape[0])

            # Sample random points
            #choices = list(range(0, points.shape[0], int(points.shape[0]/selection)))
            #points = points[choices] if points.shape[0] > selection else points
            #points.requires_grad = False

            true_quat = true[i, 8:]
            # Create a rotation matrix from quaternion
            q = F.normalize(pred[i], dim=-1)
            rot_true = mat_from_quaternion(conjugate(true_quat))[0]
            rot_pred = mat_from_quaternion(conjugate(q))[0]

            # Rotate coordinate system using rotation matrix
            m_true = torch.einsum("xj, ...j", rot_true, points)
            m_pred = torch.einsum("xj, ...j", rot_pred, points)


            #xt, yt, zt = m_true.permute(1, 0).cpu().numpy()
            #plt.clf()
            #if general:
            #    x, y, z = m_pred.permute(1, 0).detach().cpu().numpy()
            #    plt.clf()
            #    plot_points(x, y, z, figure=3, subplot=111)
            #    general = False
            #plot_points(xt, yt, zt, figure=3, subplot=122)
            #plt.show()

            err = 0
            for pp in m_pred:
                diffs = torch.pow(m_true - pp, 2)
                distances = torch.sum(diffs, dim=-1)
                err += torch.min(distances)
            batch_errs.append(err/m_pred.shape[0])
        return torch.stack(batch_errs)


class IoUAccuracy:
    def __init__(self, render_size, device, reduce=True, full=False):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device
        self.full = full

        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz.requires_grad = False


    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            a, e, t, q = torch.split(p[i], (3, 2, 3, 4))

            rot = mat_from_quaternion(conjugate(q))[0]
            t = torch.matmul(rot, t)

            coordinate_system = torch.einsum('ij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (coordinate_system[0] - t[0]) / a[0]
            y_translated = (coordinate_system[1] - t[1]) / a[1]
            z_translated = (coordinate_system[2] - t[2]) / a[2]

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

    def __call__(self, true, pred):
        #pred_quat = F.normalize(pred_quat, dim=-1)
        a = self._ins_outs(true)

        if self.full:
            b = self._ins_outs(pred)
        else: 
            b = self._ins_outs(torch.cat((true[:, :8], pred), dim=-1))

        a_bin = (a <= 1)
        b_bin = (b <= 1)

        intersection = a_bin & b_bin

        union = a_bin | b_bin
        iou = torch.sum(intersection).double()/torch.sum(union).double()

        #if iou < 0.5 or iou > 0.95:
        #    print(iou)
        #plot_render(self.xyz.cpu().numpy(), a[0].detach().cpu().numpy(), mode="in", figure=1, lims=(-1, 1))
        #plot_render(self.xyz.cpu().numpy(), b[0].detach().cpu().numpy(), mode="in", figure=2, lims=(-1, 1))
        #lt.show()

        return iou

class BinaryCrossEntropyLoss:
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
        self.xyz[self.xyz == 0] += 1e-4
        self.xyz.requires_grad = False

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            #parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(p[i], (3, 2, 3, 4))

            # Create a rotation matrix from quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            #t = torch.matmul(rot, t)
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
        a = self._ins_outs(true)
        b = self._ins_outs(torch.cat((true[:, :8], pred_quat), dim=-1))
        b_sigmoid = torch.sigmoid(5 * (1 - b))

        a[a <= 1] = 1.
        a[a > 1] = 0.
        loss = F.binary_cross_entropy(b_sigmoid, a) * 10
        
        return loss 


class AngleLoss:
    def __init__(self, device, std=0.2, reduce=True):
        self.device = device
        self.reduce = reduce

        self.std = std
        self.multi = 3

    def gx(self, x, mean):
        return self.multi * torch.exp(-0.5*((x - mean) / self.std)**2)

    def __call__(self, true, pred):
        diff = multiply(true, conjugate(pred))
        mag = to_magnitude(diff)
        loss = self.gx(mag, np.pi/2) + self.gx(mag, (3*np.pi)/2)

        if self.reduce:
            return torch.mean(loss)
        return loss

if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False



    granularity = 32
    #loss = ChamferLoss(render_size=32, device=device)

    loss2 = ChamferQuatLoss(render_size=granularity, device=device)
    acc = IoUAccuracy(render_size=128, device=device)

    torch.autograd.set_detect_anomaly(True)
    losses1 = []
    accs = []

    angles = []
    #loss_chamfer(, true_labels)
    q1b = np.array([0, 0., 0, 1 ])#randquat() #np.array([-0.34810747, -0.818965, -0.44213669, -0.11239513])
    q1e = np.array([0.9998477, 0, 0, 0.0174524])#randquat() #
    #q2 = np.array([0, 0.9998477, 0, 0.0174524])  # randquat() #
    #q2 = np.array([0, 0, 0.9998477, 0.0174524])  # randquat() #
    q2 = np.array([-0.45149812, -0.24370563, 0.78129727, -0.35543165])
    q, _ = slerp(q1b, q1e, np.arange(0, 1, 0.005))
    for q1_tmp in q:
        a1, a2, a3 = 46.23926841/255, 28.11488432/255, 66.29828836/255
        e1, e2 = 0.65549463, 0.15757771
        true = torch.tensor(
            [
                np.concatenate([[a1, a2, a3, e1, e2, 0, 0, 0], q2])

            ], device='cuda:0')
            

        pred = torch.tensor(
            [
                q1_tmp
            ],
            device='cuda:0', requires_grad=True)

        #l = l_a + l_e + l_t + l_q

        l = loss2(true, pred)
        a = acc(true, pred)

        losses1.append(l.detach().cpu().item())
        accs.append(a.detach().cpu().item())
        #losses3.append(l3.detach().cpu().item())
        
        #l1.backward()
        #l2.backward()

        #print("Grads: ", pred.grad)
        diff = multiply(torch.from_numpy(q1_tmp), conjugate(torch.from_numpy(q2)))

        #print(to_magnitude(diff))
        angles.append(to_magnitude(diff))

        # print("-----------------------------------------------")

    np_losses1 = np.array(losses1)
    np_accs = np.array(accs)

    np_angles = np.array(angles)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Angle (deg)')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(np.rad2deg(np_angles), np_losses1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=90)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.rad2deg(np_angles), np_accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(x=90)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()

    exit()



