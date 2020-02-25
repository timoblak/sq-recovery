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
from helpers import plot_render, plot_grad_flow, plot_points, norm_img, randquat
from matplotlib import pyplot as plt
from quaternion import rotate, mat_from_quaternion, conjugate


#torch.set_printoptions(sci_mode=False)



class SQNet(nn.Module):
    def __init__(self, outputs, fcn=64, dropout=0, clip_values=False):
        super(SQNet, self).__init__()
        # Parameters
        self.outputs = outputs
        self.clip_values = clip_values
        self.fcn = fcn
        self.dropout = dropout
        # Convolutions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=(3, 3))
        #nn.init.xavier_uniform(self.conv1.weight)

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
        self.fc1 = nn.Linear(128 * 16 * 16, self.fcn)
        self.fc2 = nn.Linear(self.fcn, self.outputs)

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
        self.drop = nn.Dropout(p=self.dropout)

    def forward(self, x):
        # Graph
        """
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
        """

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))



        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        #x = F.relu(self.conv5_1(x))
        #x = F.relu(self.conv5_2(x))
        #x = F.relu(self.conv5_3(x))

        """
        v = 21
        a = x[0][v].detach().cpu().numpy()
        cv2.imshow("as1d", (a - a.min()) / (a - a.min()).max())
        a = x[1][v].detach().cpu().numpy()
        cv2.imshow("as2d", (a - a.min()) / (a - a.min()).max())
        a = x[2][v].detach().cpu().numpy()
        cv2.imshow("as3d", (a - a.min()) / (a - a.min()).max())
        a = x[3][v].detach().cpu().numpy()
        cv2.imshow("as4d", (a - a.min()) / (a - a.min()).max())
        print(x[0][v])
        print(x[1][v])
        print(x[2][v])
        print(x[3][v])
        print(torch.sum(x[0][v]))
        print(torch.sum(x[1][v]))
        print(torch.sum(x[2][v]))
        print(torch.sum(x[3][v]))
        
        cv2.waitKey(0)
        """
        #for i in range(5):
        #    cv2.imshow("Image" + str(i), norm_img(cv2.resize(x[0][i].detach().cpu().numpy(), None, fx=2, fy=2)))
        #    cv2.waitKey(5)

        # Flatten + output
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


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
        #index = index % 1024
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
        #img /= 255
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
        #print("-------------------------------------------")
        #pred = F.normalize(pred, dim=-1)
        a = self._ins_outs(true)

        b = self._ins_outs(pred)
        print(a.min(), a.max())
        print(b.min(), b.max())

        #print(pred)
        x = self.xyz.cpu().numpy()
        #for i in range(1):#true.shape[0]):
        vis_a = a[0].detach().cpu().numpy()
        vis_b = b[0].detach().cpu().numpy()

        plot_render(x, vis_a, mode="in", figure=1)
        plot_render(x, vis_b, mode="in", figure=2)
        plt.show()
            #plot_render(x, np.abs(vis_a-vis_b), mode="in", figure=3)
        exit()

        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.pause(0.3)
        #plt.waitforbuttonpress(0)

        return torch.sum(torch.pow(a-b, 2)) / true.shape[0]


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
        self.xyz.requires_grad = False

    def preprocess_sq(self, p):
        a, e, t, q = torch.split(p, (3, 2, 3, 4), dim=-1)
        q = F.normalize(q, dim=-1)
        return torch.cat([a, e, t, q], dim=-1)

    def _ins_outs(self, p):
        p = p.double()
        p = self.preprocess_sq(p)
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
            #inout = torch.sqrt(inout)
            inout = torch.pow(inout, e[0])
            inout = torch.sqrt(inout)
            inout = torch.sigmoid(inout * 2 )
            #inout = A1 + B1 + C1
            results.append(inout)
        return torch.stack(results)

    def __call__(self, true, pred_quat):
        #print(true)
        #print(torch.cat((true[:, :8], pred_quat), dim=-1))
        #global  general
        pred_quat = F.normalize(pred_quat, dim=-1)
        a = self._ins_outs(true)
        b = self._ins_outs(torch.cat((true[:, :8], pred_quat), dim=-1))
        #return F.mse_loss(F.normalize(pred_quat), true[:, 8:])

        #print(a.min(), a.max())
        #print(b.min(), b.max())
        #if general:

        #    plt.clf()
        #vis_a = b[0].detach().cpu().numpy()
        #vis_b = b[1].detach().cpu().numpy()
        #vis_at = a[0].detach().cpu().numpy()
        #vis_bt = a[1].detach().cpu().numpy()
        #print(pred_quat)
        #if general:
        #    plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1, lims=(-1, 1))
        #    general = False
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2, lims=(-1, 1))
        #plot_render(self.xyz.cpu().numpy(), vis_at, mode="in", figure=3, lims=(-1, 1))
        #plot_render(self.xyz.cpu().numpy(), vis_bt, mode="in", figure=4, lims=(-1, 1))
        #   general = False
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2, lims=(-1, 1))
        #plt.show()
        #return F.mse_loss(a[0], b[0], reduction="none")

        losses = []
        for i, (ai, bi) in enumerate(zip(a, b)):

            losses.append(torch.sum(torch.pow(ai - bi, 2)))
            #losses.append(F.mse_loss(ai, bi, reduction="sum"))
        #return torch.stack(losses, dim=-1)
        #print(torch.stack(losses, dim=-1))
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
        return torch.mean(torch.stack(batch_errs))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False
    #loss = ChamferLoss(render_size=32, device=device)
    loss = ChamferQuatLoss(render_size=17, device=device)
    #loss = RotLoss(render_size=32, device=device)
    loss2 = torch.nn.MSELoss(reduction="mean")
    losses = []
    angles = []
    #loss_chamfer(, true_labels)
    q = [0, 0, 0, 1]#randquat()
    for factor in np.arange(0, 1.005, 0.005):
        angle = factor*np.pi*2
        e1, e2 = 1, 1
        a1, a2, a3 = 0.1, 0.2, 0.3

        true = torch.tensor(


            #[[ 0.1084,  0.2769,  0.1893,  0.3213,  0.7323,  0.6345,  0.5084,  0.4530, -0.6100,  0.5794, -0.5103,  0.1780]]
            [
                #[0.1569, 0.1464, 0.2784, 0.5986, 0.7872, 0.5468, 0.3737, 0.4898, 0.5681, -0.0243, -0.1486,  0.8091],
                [0.2579, 0.2211, 0.2677, 0.2644, 0.8950, 0.4700, 0.4108, 0.5803, 0, 0, 0, 1],
                [0.2579, 0.2211, 0.2677, 0.2644, 0.8950, 0.4700, 0.4108, 0.5803, 0, 0, 0, 1],
                [0.2579, 0.2211, 0.2677, 0.2644, 0.8950, 0.4700, 0.4108, 0.5803, 0, 0, 0, 1]
                #[0.2579, 0.2211, 0.2677, 0.2644, 0.8950, 0.4700, 0.4108, 0.5803, 0.9670, 0.1445, -0.1083, 0.1800]
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
                #[0.8172, -0.0995,  0.0672, -0.5637],
                #[-0.2312,  0.7598, -0.0264,  0.6071]
                [np.sin(angle / 2), 0, 0, np.cos(angle / 2)],
                [0, np.sin(angle / 2), 0, np.cos(angle / 2)],
                [0, 0, np.sin(angle / 2), np.cos(angle / 2)]

                #[0.2, 0.02, 0.5, 0.1, 0.1, 0.5, 0.5, 0, 0.042749,0.375584,0.222923,0.898563]
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

        losses.append(l.detach().cpu().numpy())
        angles.append(angle)
        #l.backward()

        print("True: " + str(true))
        print("Pred: " + str(pred))
        print("----")
        print(l)
        #print(pred.grad)
        print("-----------------------------------------------")

    np_losses = np.array(losses)
    x_axis_loss = np_losses[:, 0]
    y_axis_loss = np_losses[:, 1]
    z_axis_loss = np_losses[:, 2]


    xlimm = [0, 20]
    plt.figure(4)
    plt.subplot(131)
    plt.plot(np.rad2deg(angles), x_axis_loss)
    plt.xlabel("Angle in deg")
    plt.ylabel("Error")
    plt.title("Rotation around X axis")
    plt.ylim(xlimm)
    plt.subplot(132)
    plt.plot(np.rad2deg(angles), y_axis_loss)
    plt.xlabel("Angle in deg")
    #plt.ylabel("Error")
    plt.title("Rotation around Y axis")
    plt.ylim(xlimm)
    plt.subplot(133)
    plt.plot(np.rad2deg(angles), z_axis_loss)
    plt.xlabel("Angle in deg")
    #plt.ylabel("Error")
    plt.title("Rotation around Z axis")
    plt.ylim(xlimm)
    plt.show()
