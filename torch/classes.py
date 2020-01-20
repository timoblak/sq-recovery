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
from helpers import plot_render, plot_grad_flow
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion


torch.set_printoptions(sci_mode=False)


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

    @staticmethod
    def preprocess_sq(p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4))
        a = torch.clamp(a, min=0.05, max=1)
        e = torch.clamp(e, min=0.1, max=1)
        t = torch.clamp(t, min=0.01, max=1)
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        results = []
        for i in range(p.shape[0]):
            parameters = self.preprocess_sq(p[i])

            #print(i, parameters)
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
            E = torch.pow(torch.abs(D), (e[1] / e[0]))
            inout = E + C
            inout = torch.pow(inout, e[1])
            results.append(inout)

        return torch.stack(results)

    def __call__(self, pred, true):
        #print("-------------------------------------------")
        a = self._ins_outs(pred)
        b = self._ins_outs(true)

        #vis_a = a[0].detach().cpu().numpy()
        #vis_b = b[0].detach().cpu().numpy()

        #plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2)
        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.show()
        """
        for i, (space1, space2) in enumerate(zip(a, b)):
            print((space1 - space2).max())
            print((space1 - space2).min())

            print(i, F.l1_loss(space1, space2, reduction="mean"))

        print(F.mse_loss(a, b, reduction="mean"))
        exit()
        """
        return F.l1_loss(a, b, reduction="mean")

        #loss_val = torch.sqrt(torch.stack(list(map(torch.mean, torch.pow(diff, 2)))))
        #if self.reduce:
        #    return loss_val.mean()
        #return loss_val




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
        #self.fc2 = nn.Linear(256, self.outputs)

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
        x = self.fc1(x)

        if self.clip_values:
            x = self.clip_to_range(x)

        # Normalize quaternions
        others, q = torch.split(x, (8, 4), dim=-1)
        q = F.normalize(q)
        return others, q

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
    def __init__(self, dataset_location, labels):
        self.labels = labels
        self.dataset_location = dataset_location
        self.ext = ".bmp"
        self.h5_dataset_file = 'dataset.h5'
        self.h5_filepath = self.dataset_location #+ self.h5_dataset_file

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


if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False
    loss = ChamferLoss(26, device, reduce=True)
    loss2 = torch.nn.MSELoss(reduction="mean")

    #loss_chamfer(, true_labels)

    true = torch.tensor(
        [[0.1084,  0.2769,  0.1893,  0.3213,  0.7323,  0.6345,  0.5084,  0.4530,
        -0.6100,  0.5794, -0.5103,  0.1780]], device='cuda:0')
    pred = torch.tensor(

        #[[ 0.2415, -0.0059,  0.1914,  0.2464, -0.2308, -0.0319, -0.1229, -1.1044, 0.8736, 0.4255, 0.2318, 0.0445]],
        [[0.0500, 0.0500, 0.1128, 0.1000, 0.6181, 0.3611, 0.0100, 0.4949, 0.2348,
        0.6032, 0.2051, 0.7341]],
        device='cuda:0', requires_grad=True)


    l = loss(true, pred)
    l.backward()

    print("True: " + str(true))
    print("Pred: " + str(pred))
    print("-----------")
    print(l)
    print(pred.grad)
    print("-----------")
    plt.show()

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
