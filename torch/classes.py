import cv2
import torch
import h5py
import glob
from tqdm import tqdm
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils import data
from helpers import plot_render, plot_grad_flow
from matplotlib import pyplot as plt


class ChamferLoss:
    def __init__(self, render_size, device):
        self.render_size = render_size
        self.render_type = np.float64
        self.eps = 1e-8

        step = 1 / render_size
        #range = torch.tensor(np.arange(-self.render_size // 2, self.render_size // 2+1, 1).astype(self.render_type))
        range = torch.tensor(np.arange(0, 1+step, step).astype(self.render_type))
        xyz_list = torch.meshgrid([range, range, range])
        self.xyz = torch.stack(xyz_list).to(device)


    @staticmethod
    def preprocess_sq(p):
        a, t = torch.split(p, (3, 3))
        #e = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda:0")
        #e = torch.clamp(e, 0.1, 1)
        #t = torch.clamp(t, 0, 1)
        #a = torch.clamp(a, 0.01, 1)
        a = (a * 0.196) + 0.098
        #t = (t * 64.) + -32.
        return torch.cat([a, t], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            perameters = self.preprocess_sq(p[i])
            a, t = torch.split(perameters, (3, 3))

            # Create a rotation matrix from quaternion

            # rot = rmat.from_quaternion(q)

            # Rotate translation vector using quaternion
            # t = quat.rotate(t, q)

            # Rotate coordinate system using rotation matrix

            # m = tf.einsum('ij,jabc->iabc', rot, XYZ_GRID)

            # TWO = tf.cast(TWO, tf.float64)

            # #### Calculate inside-outside equation ####
            # First the inner parts of equation (translate + divide with axis sizes)
            x_translated = (self.xyz[0] - t[0]) / a[0]
            y_translated = (self.xyz[1] - t[1]) / a[1]
            z_translated = (self.xyz[2] - t[2]) / a[2]

            A = torch.pow(x_translated, 2)
            B = torch.pow(y_translated, 2)
            C = torch.pow(z_translated, 2)

            #A = torch.pow(A1, (1 / e[1]))# * torch.sign(x_translated)
            #B = torch.pow(B1, (1 / e[1]))  # * torch.sign(y_translated)
            #C = torch.pow(C1, (1 / e[0]))  # * torch.sign(z_translated)

            D = A + B
            E = D #torch.pow(torch.abs(D), (e[1] / e[0]))
            inout = E + C
            #inout = torch.pow(inout, e[0])
            results.append(inout)
        return torch.stack(results)

    def __call__(self, pred, true):
        a = self._ins_outs(pred)
        b = self._ins_outs(true)

        diff = a - b

        #plot_render(self.xyz.cpu().numpy(), a[0].cpu().numpy(), mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), b[0].cpu().numpy(), mode="in", figure=2)
        #plt.show()
        return torch.sqrt(torch.mean(torch.pow(diff, 2)))


class H5Dataset(data.Dataset):
    def __init__(self, dataset_location, labels):
        self.labels = labels
        self.dataset_location = dataset_location
        self.ext = ".bmp"
        self.h5_dataset_file = 'dataset.h5'

        self.dataset = None
        self.handle = None
        self.build_dataset()
        self.load_dataset()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def load_dataset(self):
        print("Opening dataset")
        self.dataset = h5py.File(self.dataset_location + self.h5_dataset_file , 'r')

    def close(self):
        self.handle.close()

    def build_dataset(self):
        if glob.glob(self.dataset_location + self.h5_dataset_file ):
            print("Using existing dataset" + str(self.dataset_location))
            return

        print("Building a new dataset " + str(self.dataset_location))
        file_list = sorted(glob.glob1(self.dataset_location, "*" + self.ext))

        with h5py.File(self.dataset_location + self.h5_dataset_file, 'w') as handle:
            ds = handle.create_dataset("sq", (len(file_list), 1, 256, 256), dtype='f')
            for i, img_name in enumerate(tqdm(file_list)):
                ds[i] = self.load_image(img_name)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Load data and get label
        X = self.dataset["sq"][index]
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
        #### First 8 for isometric model ####
        return np.concatenate([self.labels[ID][:3], self.labels[ID][5:8]])


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
        self.fc1 = nn.Linear(256 * 8 * 8, self.outputs)

        # Batch norms
        self.bn1 = nn.BatchNorm2d(32, affine= True)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Graph
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn1(self.conv2_1(x)))
        x = F.relu(self.bn1(self.conv2_2(x)))
        x = F.relu(self.bn1(self.conv2_3(x)))

        x = F.relu(self.bn2(self.conv3_1(x)))
        x = F.relu(self.bn2(self.conv3_2(x)))
        x = F.relu(self.bn2(self.conv3_3(x)))

        x = F.relu(self.bn3(self.conv4_1(x)))
        x = F.relu(self.bn3(self.conv4_2(x)))
        x = F.relu(self.bn3(self.conv4_3(x)))

        x = F.relu(self.bn4(self.conv5_1(x)))
        x = F.relu(self.bn4(self.conv5_2(x)))
        x = F.relu(self.bn4(self.conv5_3(x)))

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        if self.clip_values:
            x = self.clip_to_range(x)
        return x

    def check_grads(self):
        return
        #for p in self.parameters():
        #    print(torch.any(torch.isnan(p)))


    @staticmethod
    def clip_to_range(x):
        a, e, t = torch.split(x, (3, 2, 3), dim=-1)
        a = torch.clamp(a, min=0.01, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0.01, max=1)
        return torch.cat((a, e, t), dim=-1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False
    loss = ChamferLoss(32, device)

    #pred_p = torch.tensor([[0.2638, 0.2600, 0.6582, 0.2709, 0.56000, 1.0000, 0.4428, 0.0100]],
    #   device='cuda:0', dtype=torch.float64, requires_grad=req_grad)
    pred_p = torch.tensor([[0.1, 0.1, 0.1, 1, 1, 1, 1, 1]],
                          device='cuda:0', dtype=torch.float64, requires_grad=req_grad)
    true_p = torch.tensor([[0.0084, 0.3302, 0.0191, 0.3786, 0.7311, 0.3982, 0.3462, 0.5740]],
       device='cuda:0', dtype=torch.float64)

    #pred_p = torch.tensor([[0.3633, 0.4566, 0.4327, 0.1267, 0.1300, 0.5209, 0.3704, 0.5967]], dtype=torch.float32, requires_grad=req_grad).to(device)
    #true_p = torch.tensor([[0.3800, 0.4600, 0.4200, 0.1202, 0.1064, 0.5155, 0.3829, 0.5830]], dtype=torch.float32, requires_grad=req_grad).to(device)
    #print(torch.autograd.gradcheck(loss.__call__, (true_p, pred_p)))
    l = loss(true_p, pred_p)

    print(l)
    exit()

    net = SQNet(8).to(device)
    summary(net, (1, 256, 256))

    a = cv2.imread("../data/data_iso/000000.bmp", 0)
    a = np.expand_dims(a, -1)
    a = np.transpose(a, (2, 0, 1))

    my_img_tensor = torch.from_numpy(a)
    my_img_tensor = my_img_tensor.type('torch.DoubleTensor')
    my_img_tensor *= (1/255)
    my_img_tensor = my_img_tensor.unsqueeze(0)
    my_img_tensor.to(device)
    my_img_tensor = my_img_tensor.type('torch.FloatTensor')

    print(net.forward(my_img_tensor))
