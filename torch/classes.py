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
from helpers import plot_render, plot_grad_flow, plot_points
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion


torch.set_printoptions(sci_mode=False)



class SQNet(nn.Module):
    def __init__(self, outputs, clip_values=False):
        super(SQNet, self).__init__()
        # Parameters
        self.outputs = outputs
        self.clip_values = clip_values

        # Convolutions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=(3, 3))

        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1))
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=(1, 1))

        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=(1, 1))

        # Fully connected
        self.fc1 = nn.Linear(256 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, self.outputs)

        # Batch norms
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn1_3 = nn.BatchNorm2d(32)
        self.bn1_4 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn4_3 = nn.BatchNorm2d(256)

        # Dropout
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # Graph
        x = F.relu(self.bn1_1(self.conv1(x)))

        x = F.relu(self.bn1_2(self.conv2_1(x)))
        x = F.relu(self.bn1_3(self.conv2_2(x)))
        x = F.relu(self.bn1_4(self.conv2_3(x)))

        x = F.relu(self.bn2_1(self.conv3_1(x)))
        x = F.relu(self.bn2_2(self.conv3_2(x)))
        x = F.relu(self.bn2_3(self.conv3_3(x)))

        x = F.relu(self.bn3_1(self.conv4_1(x)))
        x = F.relu(self.bn3_2(self.conv4_2(x)))
        x = F.relu(self.bn3_3(self.conv4_3(x)))

        x = F.relu(self.bn4_1(self.conv5_1(x)))
        x = F.relu(self.bn4_2(self.conv5_2(x)))
        x = F.relu(self.bn4_3(self.conv5_3(x)))

        # Flatten + output
        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.clip_values:
            x = self.clip_to_range(x)

        # Normalize quaternions
        #others, q = torch.split(x, (8, 4), dim=-1)

        return x

    @staticmethod
    def clip_to_range(x):
        a, e, t, q = torch.split(x, (3, 2, 3, 4), dim=-1)
        a = torch.clamp(a, min=0.01, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0.01, max=1)
        return torch.cat((a, e, t, q), dim=-1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class H5Dataset(data.Dataset):
    def __init__(self, dataset_location, labels, dataset_file='dataset.h5'):
        self.labels = labels
        self.dataset_location = dataset_location
        self.ext = ".bmp"
        self.h5_dataset_file = dataset_file
        self.h5_filepath = self.dataset_location + self.h5_dataset_file

        self.dataset = None
        self.handle = None
        self.build_dataset()
        #self.load_dataset()

    def __len__(self):
        return len(self.labels)

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
        #index = 0
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
        img /= 255
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

        step = 1 / render_size
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.xyz.requires_grad = False

    @staticmethod
    def preprocess_sq(p):
        #a, t, q = torch.split(p, (3, 3, 4))
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0.01, max=1)
        #return torch.cat([a, t, q], dim=-1)
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            parameters = self.preprocess_sq(p[i])
            a, e, t, q = torch.split(parameters, (3, 2, 3, 4))

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
            E = torch.pow(D, (e[1] / e[0]))
            inout = E + C
            #inout = torch.pow(inout, e[0])
            #inout = A1 + B1 + C1

            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred):
        #print("-------------------------------------------")
        pred = F.normalize(pred, dim=-1)
        a = self._ins_outs(true)
        b = self._ins_outs(pred)
        #print(pred)
        #x = self.xyz.cpu().numpy()
        #for i in range(true.shape[0]):
            #vis_a = a[i].detach().cpu().numpy()
            #vis_b = b[i].detach().cpu().numpy()

            #plot_render(x, vis_a, mode="in", figure=1)
            #plot_render(x, vis_b, mode="in", figure=2)
            #plot_render(x, np.abs(vis_a-vis_b), mode="in", figure=3)

        #plt.draw()
        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.pause(0.3)
        #plt.waitforbuttonpress(0)

        return torch.sum(torch.pow(a-b, 2)) / true.shape[0]


class ChamferQuatLoss:
    def __init__(self, render_size, device, reduce=True):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8
        self.reduce = reduce

        step = 2 / render_size
        range = torch.tensor(np.arange(-1, 1 + step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)
        self.device = device

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4), dim=-1)
        a = a * 2
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = self.preprocess_sq(p)
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            a, e, t, q = torch.split(p[i], (3, 2, 3, 4))

            q = F.normalize(q, dim=-1)
            # Create a rotation matrix from quaternion
            rot = mat_from_quaternion(q)

            # Rotate coordinate system using rotation matrix
            m = torch.einsum('xij,jabc->iabc', rot, self.xyz)

            #m = rotate(self.xyz.permute(1, 2, 3, 0), q).permute(3, 0, 1, 2)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
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
            inout = torch.pow(inout, e[1])
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred_quat):
        a = self._ins_outs(true)
        true_block, true_quat = torch.split(true, (8, 4), dim=-1)
        b = self._ins_outs(torch.cat((true_block, pred_quat), dim=-1))

        #vis_a = a[0].detach().cpu().numpy()
        #vis_b = b[0].detach().cpu().numpy()
        #plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1, lims=(-1, 1))
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2, lims=(-1, 1))
        #plt.show()
        return F.mse_loss(a, b, reduction="mean")
        losses = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            pw = torch.pow(ai - bi, 2)
            diff = torch.sum(pw)

            losses.append(diff)

        return torch.mean(torch.stack(losses, dim=-1))


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

        batch_errs = []
        for i in range(true.shape[0]):
            # Gererate the inout space
            inout = self._ins_outs(true[i])

            # determine the border around surface, that points are sampled from
            #t = time()
            for border in np.arange(0, 1, 0.01):
                indxs = torch.where((inout < (1 + border)) & (inout > (1 - border)))
                if indxs[0].shape[0] > 50:
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
            rot_true = mat_from_quaternion(true_quat)[0]
            rot_pred = mat_from_quaternion(q)[0]

            # Rotate coordinate system using rotation matrix
            m_true = torch.einsum("xj, ...j", rot_true, points)
            m_pred = torch.einsum("xj, ...j", rot_pred, points)

            #x, y, z = m_pred.permute(1, 0).detach().cpu().numpy()
            #xt, yt, zt = m_true.permute(1, 0).cpu().numpy()
            #plt.clf()
            #plot_points(x, y, z, figure=3, subplot=121)
            #plot_points(xt, yt, zt, figure=3, subplot=122)
            #plt.show()

            err = 0
            for pp in m_pred:
                diffs = torch.pow(m_true - pp, 2)
                distances = torch.sum(diffs, dim=-1)
                err += torch.min(distances)
            batch_errs.append(err)

        #exit()
        return torch.mean(torch.stack(batch_errs))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False
    loss = ChamferQuatLoss(render_size=15, device=device)
    #loss = RotLoss(render_size=32, device=device)
    loss2 = torch.nn.MSELoss(reduction="mean")
    losses = []
    #loss_chamfer(, true_labels)
    for factor in np.arange(0, 1.005, 0.005):
        angle = factor*np.pi*2
        true = torch.tensor(

            #[[ 0.1084,  0.2769,  0.1893,  0.3213,  0.7323,  0.6345,  0.5084,  0.4530, -0.6100,  0.5794, -0.5103,  0.1780]]
            [
                #[0.1, 0.1, 0.1, 0.1, 0.1,  0.6345,  0.5084,  0.4530, 0, 0, 0, 1.],
                [0.2390,  0.2537,  0.2847,  0.5455,  0.8367,  0.5167,  0.3494,  0.4659,
          0.0301,  0.0014, -0.9108, -0.4117],
                #[0.1,  0.1,  0.10,  0.154,  0.1,  0.3786,  0.4682,  0.3593, -0.5879, -0.5388, -0.4251, -0.4282],
                #[0, 0, 0, 1.]
            ], device='cuda:0')
            #[[0.1084, 0.2769, 0.1893, 0.6345, 0.5084, 0.4530, -0.6100, 0.5794, -0.5103, 0.1780]]

        pred = torch.tensor(
            #[[ 0.2415, -0.0059,  0.1914,  0.2464, -0.2308, -0.0319, -0.1229, -1.1044, 0.8736, 0.4255, 0.2318, 0.0445]],
            #[[0.1084, 0.2769, 0.1893, 0.2517, 0.5858, 0.6345, 0.5084, 0.4530, 0.1782,0.5103, 0.5794, 0.6100]],
            #[[0.0500, 0.0500, 0.1128, 0.3611, 0.0100, 0.4949, 0.2348, 0.6032, 0.2051, 0.7341]],
            #[[ 0.7071068, 0, 0, 0.7071068 ]],

            [
                #[np.sin(angle / 2), 0, 0, np.cos(angle / 2)],
                [0.0467, 0.1050, 0.1301, 0.0061]
            ],
            device='cuda:0', requires_grad=True)

        #a, e, t, q = torch.split(pred, (3, 2, 3, 4), dim=-1)
        #a_true, e_true, t_true, q_true = torch.split(true, (3, 2, 3, 4), dim=-1)

        #l_a = loss(true, torch.cat((a, e_true, t_true, q_true), dim=-1))
        #l_e = loss(true, torch.cat((a_true, e, t_true, q_true), dim=-1))
        #l_t = loss(true, torch.cat((a_true, e_true, t, q_true), dim=-1))
        #l_q = loss(true, torch.cat((a_true, e_true, t_true, q), dim=-1))

        #l = l_a + l_e + l_t + l_q
        l = loss(true, pred)
        losses.append(l.item())
        l.backward()

        print("True: " + str(true))
        print("Pred: " + str(pred))
        print("----")
        print(l)
        print(pred.grad)
        print("-----------------------------------------------")
        break

    plt.plot(losses)
    plt.show()
