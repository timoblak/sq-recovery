import cv2
import torch
import h5py
import glob
from helpers import gray_to_jet
from time import time
from tqdm import tqdm
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils import data
from helpers import plot_render, plot_grad_flow, plot_points, norm_img, randquat, slerp, parse_csv
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
        #index = index % 1
        if self.mode == 1:
            index += self.n_train
        #print("Fetching index", index)
        with h5py.File(self.h5_filepath, 'r') as db:
            X = db["sq"][index]

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
        # ### First 8 for isometric model, 12 for full model ####
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


class ExplicitLoss:
    """
    Explicit loss class:
    -> calculates loss as MSE between occupancy functions of predicted and true parameters.
    """
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device

        # Setup discretized space
        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz[self.xyz == 0] += 1e-4
        self.xyz.requires_grad = False

    @staticmethod
    def preprocess_sq(p):
        # Clamp parameters into ranges, suitable for calculating the inside-outside function
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0, max=1)
        return torch.cat([a, e, t, q], dim=-1)

    def occupancy(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(parameters, (3, 2, 3, 4))

            # To transform info general space either:
            # - contruct 4x4 homogenous transformation matrix from q and t.
            #   then inverse and apply to xyz space, or
            # - take conjugate of q (or transpose of rotation matrix) and then
            #   first rotate translation vector, then rotate and translate the space <- IMPLEMENTED

            # Create a rotation matrix from conjugated quaternion
            rot = mat_from_quaternion(conjugate(q))[0]

            # Rotate translation vector
            t = torch.matmul(rot, t)

            # Rotate coordinate space
            coordinate_system = torch.einsum('ij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (coordinate_system[0] - t[0]) / a[0]
            y_translated = (coordinate_system[1] - t[1]) / a[1]
            z_translated = (coordinate_system[2] - t[2]) / a[2]

            # Calculate powers of 2 to get rid of negative values)
            # + add small number for stability (we don't want zeros going into pow())
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

            # Final power, which results in an equaliy-represented parameter space for e1 and e2
            inout = torch.pow(inout, e[0])

            # Activation function which converts the inside-outside function into an occupancy function
            inout = torch.sigmoid(5 * (1 - inout))
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred):
        a = self.occupancy(true)
        b = self.occupancy(pred)

        losses = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            # Multiplier is added to increase gradients in parameter space. Could also probably just increase LR
            l = torch.mean(torch.pow(ai - bi, 2)) * 100
            losses.append(l)
        ls = torch.mean(torch.stack(losses, dim=-1))
        return ls

class ImplicitLoss:
    """
    Implicit loss class:
    -> calculates loss as MAE between projected depth image of predicted parameters and input depth image
    """
    def __init__(self, render_size, device, tau=1, sigmoid_sharpness=100, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device
        self.tau = tau
        self.sigmoid_sharpness = sigmoid_sharpness

        # Create discretized space
        range = torch.tensor(np.linspace(0, 1, self.render_size).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz[self.xyz == 0] += 1e-4
        self.xyz.requires_grad = False

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0, max=1)
        return torch.cat([a, e, t, q], dim=-1)

    def depth_projection(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(parameters, (3, 2, 3, 4))

            # To transform info general space either:
            # - contruct 4x4 homogenous transformation matrix from q and t.
            #   then inverse and apply to xyz space
            # - take conjugate of q (or transpose of rotation matrix) and then
            #   first rotate translation vector, then rotate and translate the space <- IMPLEMENTED

            # Create a rotation matrix from conjugated quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            tr = torch.matmul(rot, t)

            coordinate_system = torch.einsum('ij,jabc->iabc', rot, self.xyz)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (coordinate_system[0] - tr[0]) / a[0]
            y_translated = (coordinate_system[1] - tr[1]) / a[1]
            z_translated = (coordinate_system[2] - tr[2]) / a[2]

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

            # Activation function which converts the inside-outside function into an occupancy function
            inout = torch.pow(inout, e[0])
            inout = torch.sigmoid(self.sigmoid_sharpness * (1 - inout))

            # Calculate the depth projection
            cumulative_depth = torch.exp(-self.tau * torch.cumsum(inout.flip(dims=[-1]), dim=-1))
            depth = 1 - cumulative_depth.sum(dim=-1) / self.render_size  # torch.exp(-cs_depth.sum(dim=-1))
            depth = depth.permute((1, 0)).flip(dims=(0,))

            results.append(depth)
        return torch.stack(results)

    def __call__(self, true, pred):
        # Resize the depth image to (render_size x render_size)
        true_resized = F.interpolate(true, size=(self.render_size, self.render_size), mode="nearest")

        # Calculate the
        pred_depths = self.depth_projection(pred).unsqueeze(1)
        losses = []
        for i, (ai, bi) in enumerate(zip(true_resized, pred_depths)):
            l = torch.mean(torch.abs(ai - bi))
            losses.append(l)
        ls = torch.mean(torch.stack(losses, dim=-1))
        return ls

class LeastSquares:
    """
    Least squares loss class:
    -> loss is minimization of "distance" between real SQ points (from depth image) and predicted SQ surface
    -> based on Solina & Bajcsy's original iterative minimization
    """
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0, max=1)
        return torch.cat([a, e, t, q], dim=-1)

    def energy_function(self, batch_points, params):
        # params = params.double()
        results = []
        for i in range(params.shape[0]):
            parameters = self.preprocess_sq(params[i])
            a, e, t, q = torch.split(parameters, (3, 2, 3, 4))
            points = batch_points[i]

            # Create a rotation matrix from conjugated quaternion
            rot = mat_from_quaternion(conjugate(q))[0]
            tr = torch.matmul(rot, t)

            rotated_points = torch.einsum('ij,ja->ia', rot, points)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (rotated_points[0] - tr[0]) / a[0]
            y_translated = (rotated_points[1] - tr[1]) / a[1]
            z_translated = (rotated_points[2] - tr[2]) / a[2]

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

            # Use the original energy function as final loss
            inout = torch.pow(torch.sqrt(a[0] * a[1] * a[2]) * (torch.pow(inout, e[0]) - 1), 2)
            results.append(inout.sum())
        return torch.stack(results)

    def __call__(self, true, pred):
        true_resized = F.interpolate(true, size=(self.render_size, self.render_size), mode="nearest")

        batch_points = []
        for i in range(true.shape[0]):
            xy = torch.where(true_resized[i][0] > 0)

            x = xy[0].float() / self.render_size
            y = xy[1].float() / self.render_size
            z = true_resized[i][0][xy]
            points = torch.stack([y, 1-x, z])  # .permute((1, 0))
            batch_points.append(points)
        loss = self.energy_function(batch_points, pred)
        return loss.mean()


class IoUAccuracy:
    """
    IoU accuracy:
    -> calculates the IoU between binerized predicted and true superquadrics
    -> higher the render_size, the more accurate estimation is (use at least 64)
    """

    def __init__(self, render_size, device, reduce=True, full=False):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce
        self.device = device
        self.full = full

        range = torch.tensor(np.linspace(0, 1, render_size).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz.requires_grad = False

    def ins_outs(self, p):
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
        a = self.ins_outs(true)
        b = self.ins_outs(pred)

        # Binarize inside-outside equation
        a_bin = (a <= 1)
        b_bin = (b <= 1)

        # Calculate IoU
        intersection = a_bin & b_bin
        union = a_bin | b_bin
        iou = torch.sum(intersection) / torch.sum(union)

        if not self.reduce:
            ious = []
            for i in range(union.shape[0]):
                ious.append(torch.sum(intersection[i]).double()/torch.sum(union[i]).double())
            return torch.stack(ious)

        return iou





if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False

    acc = IoUAccuracy(render_size=64, device=device)
    a1, a2, a3, e1, e2 = 28.985552/255, 61.850255/255, 68.976172/255, 0.215097, 0.275022
    t1, t2, t3 = 137.818167/255, 94.702536/255, 118.771105/255
    q1 = np.array([0.699625,0.378123,-0.090419,-0.599476])
    q2 = np.array([0.699625,0.378123,-0.090419,-0.599476])

    true = torch.tensor([
            np.concatenate([[a1, a2, a3, e1, e2, t1, t2, t3], q1])
        ], device='cuda:0')

    pred = torch.tensor([
            np.concatenate([[a1, a2, a3, e1, e2, t1, t2, t3], q2])
        ],
        device='cuda:0', requires_grad=True)


    a = acc(true, pred)


    """

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



"""