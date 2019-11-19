import cv2
import torch
import numpy as np
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils import data
from helpers import plot_render

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
        a, e, t = torch.split(p, (3, 2, 3))
        a = (a * 0.196) + 0.098
        #e = torch.clamp(e, 0.1, 1)
        #t = (t * 64.) + -32.
        return torch.cat([a, e, t], dim=-1)

    def _ins_outs(self, p):

        p = p.double()
        results = []
        for i in range(p.shape[0]):
            perameters = self.preprocess_sq(p[i])
            a, e, t= torch.split(perameters, (3, 2, 3))

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
            # Then calculate all of the powers (make sure to calculate power over absolute values to avoid complex numbers)
            A = torch.pow(torch.abs(x_translated), (2 / e[1]))#*torch.sign(x_translated)
            B = torch.pow(torch.abs(y_translated), (2 / e[1]))#*torch.sign(y_translated)
            C = torch.pow(torch.abs(z_translated), (2 / e[0]))#*torch.sign(z_translated)
            D = A + B
            E = torch.pow(torch.abs(D), (e[1] / e[0]))
            inout = E + C

            results.append(inout)
        return torch.stack(results)

    def __call__(self, pred, true):
        start_t = time()
        a = self._ins_outs(pred)
        b = self._ins_outs(true)

        #diff = torch.log(a + self.eps) - torch.log(b + self.eps)
        diff = a - b
        #print("Time: " + str(time() - start_t))

        #plot_render(self.xyz.cpu().numpy(), a[0].cpu().numpy(), mode="in")
        #plot_render(self.xyz.cpu().numpy(), b[0].cpu().numpy(), mode="in")
        #print(a)
        #exit()
        return torch.sqrt(torch.mean(torch.pow(diff, 2)))


class Dataset(data.Dataset):
    def __init__(self, dataset_location, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs
        self.dataset_location = dataset_location
        self.ext = ".bmp"

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.load_image(ID)
        y = self.load_label(ID)

        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

    def load_image(self, ID):
        # Load image and reshape to (channels, height, width)
        img = cv2.imread(self.dataset_location + ID + self.ext, 0).astype(np.float)
        img = np.expand_dims(img, -1)
        img = np.transpose(img, (2, 0, 1))
        # Do preprocessing
        img /= 255
        return img

    def load_label(self, ID):
        # Select labels and preprocess
        return self.labels[ID][:8] #### First 8 for isometric model ####


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

    @staticmethod
    def clip_to_range(x):
        a, e, t = torch.split(x, (3, 2, 3), dim=-1)
        a = torch.clamp(a, min=0, max=1)
        e = torch.clamp(e, min=0.01, max=1)
        t = torch.clamp(t, min=0, max=1)
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
    loss = ChamferLoss(16, device)
    #25.000000, 49.000000, 55.000000, 0.084488, 0.751716, 99.890729, 164.551254, 155.331333
    pred_p = torch.tensor([[0.8184, 0.4989, 0.9407, 0.6035, 0.2144, 0.3787, 0.5676, 0.6420]], dtype=torch.float32, requires_grad=req_grad).to(device)
    true_p = torch.tensor([[0.8200, 0.5000, 0.9400, 0.6019, 0.1960, 0.3890, 0.5606, 0.6412]], dtype=torch.float32, requires_grad=req_grad).to(device)
    print(pred_p.requires_grad)
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