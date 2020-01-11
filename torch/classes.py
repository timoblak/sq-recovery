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

        #vis_a = a[0].detach().cpu().numpy()
        #vis_b = b[0].detach().cpu().numpy()

        #plot_render(self.xyz.cpu().numpy(), vis_a, mode="in", figure=1)
        #plot_render(self.xyz.cpu().numpy(), vis_b, mode="in", figure=2)
        #vis_a[vis_a <= 1] = 1; vis_a[vis_a > 1] = 0
        #vis_b[vis_b <= 1] = 1; vis_b[vis_b > 1] = 0
        #iou = np.bitwise_and(vis_a.astype(bool), vis_b.astype(bool))
        #plot_render(self.xyz.cpu().numpy(), iou.astype(int), mode="bit", figure=3)
        #plt.show()

        loss_val = torch.sqrt(torch.stack(list(map(torch.mean, torch.pow(diff, 2)))))
        if self.reduce:
            return loss_val.mean()
        return loss_val


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
        self.bn1 = nn.BatchNorm2d(32)
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

        # Flatten + output
        x = x.view(-1, self.num_flat_features(x))
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
        q = q / torch.norm(q)
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


    pred = torch.cat((torch.tensor([[0.3347, 0.5088, 0.6226, 0.5490, 0.5938, 0.6148, 0.5062, 0.4543],
             [0.2863, 0.3303, 0.3909, 0.5667, 0.5277, 0.4243, 0.4548, 0.4140],
             [0.3649, 0.6342, 0.4868, 0.3958, 0.5441, 0.4743, 0.5816, 0.5610],
             [0.5847, 0.4456, 0.7256, 0.2993, 0.5417, 0.6037, 0.4486, 0.4790],
             [0.2564, 0.2956, 0.2633, 0.6782, 0.5424, 0.4552, 0.6131, 0.4246],
             [0.6422, 0.7043, 0.7510, 0.4678, 0.4724, 0.4139, 0.5326, 0.4195],
             [0.4677, 0.4156, 0.5068, 0.3734, 0.5042, 0.6987, 0.4509, 0.4812],
             [0.4836, 0.4193, 0.6916, 0.2560, 0.5217, 0.3665, 0.6246, 0.4488],
             [0.4325, 0.4963, 0.4454, 0.5434, 0.3748, 0.4235, 0.5239, 0.6774],
             [0.4232, 0.5344, 0.5961, 0.6730, 0.5547, 0.6193, 0.5166, 0.5825],
             [0.2689, 0.4771, 0.3442, 0.5892, 0.5460, 0.5442, 0.6559, 0.6398],
             [0.4558, 0.5988, 0.5671, 0.4852, 0.5834, 0.5180, 0.5783, 0.4093],
             [0.3275, 0.7032, 0.5445, 0.5925, 0.6850, 0.5690, 0.6034, 0.4710],
             [0.2942, 0.4801, 0.5660, 0.4696, 0.4707, 0.5731, 0.4632, 0.6674],
             [0.4103, 0.5005, 0.5618, 0.5071, 0.4896, 0.3676, 0.4314, 0.4968],
             [0.2990, 0.3597, 0.2838, 0.4894, 0.5223, 0.6129, 0.5628, 0.3825],
             [0.1642, 0.1690, 0.2933, 0.6742, 0.5554, 0.4544, 0.5148, 0.3631],
             [0.1542, 0.2128, 0.2375, 0.6986, 0.6514, 0.4664, 0.6354, 0.5124],
             [0.5412, 0.4533, 0.5145, 0.5288, 0.4234, 0.6512, 0.5040, 0.6977],
             [0.3635, 0.3819, 0.4178, 0.4486, 0.5668, 0.4104, 0.3967, 0.5392],
             [0.6602, 0.9072, 0.6615, 0.3426, 0.5609, 0.6386, 0.5150, 0.6687],
             [0.4651, 0.6809, 0.5833, 0.3948, 0.4528, 0.5257, 0.3383, 0.4114],
             [0.4359, 0.7260, 0.6854, 0.6872, 0.6718, 0.5966, 0.4026, 0.4258],
             [0.2345, 0.3431, 0.3362, 0.6026, 0.5530, 0.3758, 0.5081, 0.4210],
             [0.3316, 0.5693, 0.4524, 0.5301, 0.5332, 0.5109, 0.4468, 0.2954],
             [0.5637, 0.5551, 0.4874, 0.5728, 0.5788, 0.5358, 0.5929, 0.3554],
             [0.4756, 0.6051, 0.5838, 0.4204, 0.6063, 0.6412, 0.4509, 0.2445],
             [0.3548, 0.2603, 0.5326, 0.3894, 0.6312, 0.5584, 0.3390, 0.4479],
             [0.3591, 0.8149, 0.4888, -0.0580, 0.6377, 0.4158, 0.6134, 0.6338],
             [0.4988, 0.4668, 0.5954, 0.6022, 0.5817, 0.4910, 0.6823, 0.5826],
             [0.3411, 0.4094, 0.6084, 0.6967, 0.4585, 0.5347, 0.4634, 0.5308],
             [0.2066, 0.2884, 0.2329, 0.6051, 0.7854, 0.6659, 0.3639, 0.4176]],
            device='cuda:0'), torch.tensor([[-0.0080, 0.8317, 0.0769, -0.5498],
               [0.1782, 0.7561, -0.0907, -0.6232],
               [-0.0331, 0.8794, -0.1499, -0.4507],
               [-0.2477, 0.8961, -0.1993, -0.3097],
               [-0.2531, 0.8559, 0.2485, -0.3764],
               [-0.1250, 0.4895, 0.3188, -0.8020],
               [-0.3351, 0.8550, -0.2805, -0.2794],
               [0.0376, 0.5780, 0.0353, -0.8144],
               [-0.1886, 0.6036, 0.3476, -0.6923],
               [-0.2567, 0.8666, -0.1331, -0.4067],
               [-0.1439, 0.7423, -0.2968, -0.5833],
               [-0.1165, 0.8361, -0.0630, -0.5324],
               [0.0874, 0.7528, 0.1554, -0.6336],
               [-0.3093, 0.8443, -0.2949, -0.3232],
               [-0.0781, 0.5878, 0.3275, -0.7356],
               [-0.0033, 0.6705, 0.1638, -0.7236],
               [0.1503, 0.8152, 0.2110, -0.5179],
               [-0.1874, 0.8666, 0.2388, -0.3961],
               [-0.3349, 0.7551, 0.0560, -0.5608],
               [0.2419, 0.4812, 0.0994, -0.8367],
               [-0.0593, 0.6825, -0.0124, -0.7283],
               [0.0890, 0.7791, 0.2034, -0.5863],
               [-0.1750, 0.8016, 0.0107, -0.5716],
               [0.0596, 0.7464, -0.1974, -0.6328],
               [0.1871, 0.7409, 0.0548, -0.6427],
               [0.1949, 0.8489, 0.2154, -0.4416],
               [0.0948, 0.6068, -0.1361, -0.7774],
               [-0.0588, 0.7006, -0.4422, -0.5569],
               [-0.3045, 0.7196, -0.2037, -0.5898],
               [0.0109, 0.6663, -0.1488, -0.7306],
               [0.0532, 0.9007, 0.0494, -0.4282],
               [-0.1012, 0.5740, -0.1295, -0.8022]],
              device='cuda:0')), dim=-1)
    pred = torch.autograd.Variable(pred, requires_grad=True)
    true = torch.tensor([[8.3187e-01, 2.5209e-01, 4.6577e-01, 4.9967e-01, 7.5574e-01,
             6.5489e-01, 4.9402e-01, 3.9512e-01, -8.7377e-01, 3.9702e-01,
             -2.7770e-01, 4.2208e-02],
            [7.5169e-02, 8.1225e-01, 5.3486e-01, 9.8933e-01, 7.4691e-01,
             3.7630e-01, 4.2227e-01, 3.6366e-01, -8.6963e-01, 3.8346e-01,
             -1.7720e-01, 2.5554e-01],
            [3.1576e-01, 5.2318e-01, 4.8603e-01, 2.2613e-01, 6.2383e-01,
             4.9835e-01, 5.8971e-01, 6.3394e-01, -9.8400e-04, -5.4293e-01,
             3.1763e-01, -7.7739e-01],
            [1.3883e-01, 4.8413e-01, 8.5914e-01, 6.1708e-01, 6.7608e-01,
             6.1266e-01, 4.3551e-01, 6.4695e-01, -6.1855e-01, 8.8051e-02,
             -7.3481e-01, -2.6399e-01],
            [4.0356e-01, 2.9589e-02, 2.2282e-01, 5.7177e-01, 8.6336e-01,
             3.9245e-01, 6.3430e-01, 4.6110e-01, 9.4003e-01, -1.9009e-01,
             1.8088e-01, 2.1791e-01],
            [9.5795e-01, 3.8770e-01, 5.7179e-01, 7.0822e-01, 2.2876e-01,
             3.9366e-01, 5.3501e-01, 4.5646e-01, -2.3366e-01, 8.8611e-01,
             -3.1477e-01, -2.4726e-01],
            [3.7605e-01, 1.8969e-01, 8.0906e-01, 6.8860e-01, 3.5507e-01,
             6.4031e-01, 4.0401e-01, 5.7740e-01, 2.3533e-01, -5.9479e-01,
             7.6832e-01, 2.3047e-02],
            [6.8750e-01, 2.9501e-01, 4.1817e-01, 2.3352e-01, 1.4925e-01,
             3.7553e-01, 5.9335e-01, 5.2072e-01, -4.2268e-01, -4.8841e-01,
             -3.7257e-01, -6.6632e-01],
            [6.7114e-01, 4.5510e-01, 2.9813e-01, 4.7348e-01, 3.8865e-01,
             3.8807e-01, 5.1572e-01, 6.1664e-01, -5.4015e-01, -2.1035e-01,
             4.6409e-01, 6.6979e-01],
            [7.1392e-01, 6.2187e-01, 8.0151e-01, 5.8600e-01, 9.8527e-01,
             5.6822e-01, 5.5164e-01, 5.5478e-01, 1.5715e-01, 9.6290e-01,
             2.1467e-01, -4.5225e-02],
            [4.0327e-01, 3.2465e-01, 1.5162e-01, 5.3244e-01, 1.1595e-01,
             4.9790e-01, 6.0572e-01, 6.4329e-01, -5.4145e-01, -7.6454e-01,
             -3.1323e-01, 1.5556e-01],
            [2.0535e-01, 6.3083e-01, 7.8227e-01, 3.5893e-01, 5.7847e-01,
             4.6643e-01, 5.4834e-01, 5.0693e-01, 4.9490e-02, -2.1780e-01,
             5.8507e-01, -7.7961e-01],
            [8.6417e-01, 4.1994e-01, 5.2892e-01, 9.4574e-01, 6.0639e-01,
             5.7792e-01, 6.3911e-01, 5.0770e-01, 6.2749e-01, 5.4344e-01,
             -1.2547e-01, 5.4331e-01],
            [5.9791e-01, 1.9807e-01, 3.7519e-01, 5.0754e-01, 1.7595e-01,
             6.3467e-01, 4.4176e-01, 6.3294e-01, 7.1094e-01, 2.7203e-01,
             1.9493e-01, 6.1852e-01],
            [8.8639e-01, 1.6596e-01, 3.4620e-01, 7.1779e-01, 1.5974e-01,
             3.5280e-01, 4.9650e-01, 4.7159e-01, 8.2047e-01, 1.3454e-01,
             4.0361e-01, 3.8187e-01],
            [3.4742e-01, 4.3935e-02, 3.7677e-01, 5.5693e-01, 6.1049e-01,
             6.3659e-01, 5.8471e-01, 3.7780e-01, 2.7257e-01, 1.4088e-01,
             -3.8289e-01, 8.7135e-01],
            [6.9881e-02, 1.3444e-01, 2.3257e-01, 4.7895e-01, 8.0102e-01,
             4.5825e-01, 5.4777e-01, 3.6133e-01, -3.4097e-01, -6.2734e-01,
             5.1523e-01, 4.7405e-01],
            [1.1745e-01, 1.6811e-01, 1.4423e-02, 8.4182e-01, 6.5215e-01,
             4.1968e-01, 6.4456e-01, 5.9445e-01, -4.7878e-01, 5.0667e-01,
             4.0107e-01, 5.9430e-01],
            [3.3020e-01, 4.6760e-01, 7.5497e-01, 4.4951e-01, 6.2829e-01,
             4.6963e-01, 4.5500e-01, 6.5464e-01, 7.4436e-01, 1.4386e-01,
             6.5072e-01, 4.2289e-02],
            [3.3914e-01, 9.9193e-02, 3.0277e-01, 1.3219e-01, 3.3382e-01,
             3.6668e-01, 3.7329e-01, 5.8527e-01, -2.8670e-01, 4.6114e-01,
             -6.1341e-01, 5.7348e-01],
            [9.0201e-01, 9.8365e-01, 3.6541e-01, 3.2634e-01, 6.7608e-01,
             4.9329e-01, 5.1244e-01, 6.5075e-01, -7.4926e-02, -2.6216e-01,
             9.6210e-01, 4.5050e-03],
            [2.1768e-01, 8.7994e-01, 7.8483e-01, 2.0133e-01, 7.5466e-01,
             5.7219e-01, 3.8731e-01, 3.8836e-01, 2.0782e-01, -9.1027e-02,
             6.4796e-02, 9.7176e-01],
            [5.1874e-01, 5.9763e-01, 2.2962e-01, 4.5195e-01, 6.1385e-01,
             6.0670e-01, 3.6872e-01, 4.5375e-01, -5.4270e-01, -6.3103e-01,
             2.3809e-01, -5.0059e-01],
            [4.0195e-01, 8.0732e-02, 9.1379e-01, 9.0841e-01, 6.4250e-01,
             3.8658e-01, 4.5084e-01, 3.5625e-01, -9.5374e-01, -2.5464e-01,
             1.4821e-01, -5.9753e-02],
            [8.0528e-01, 4.9859e-01, 3.3601e-02, 2.9695e-01, 6.2431e-01,
             5.0615e-01, 4.6950e-01, 3.4729e-01, 3.7499e-01, -4.6795e-01,
             -8.0014e-01, 1.3437e-02],
            [4.6244e-01, 9.5340e-01, 3.9564e-01, 7.6872e-01, 5.5670e-01,
             4.8871e-01, 5.7752e-01, 3.4869e-01, -4.4877e-01, -8.0548e-02,
             1.7511e-02, -8.8984e-01],
            [3.1446e-01, 7.2259e-01, 6.0716e-01, 5.0525e-01, 9.5039e-01,
             6.4237e-01, 5.2894e-01, 3.6264e-01, 2.6937e-01, -8.1435e-01,
             7.2790e-02, 5.0890e-01],
            [5.4733e-01, 5.4532e-01, 3.9047e-01, 4.4240e-01, 3.5223e-01,
             5.1604e-01, 3.5588e-01, 5.5793e-01, 1.6284e-01, -7.3045e-01,
             2.0018e-01, 6.3234e-01],
            [3.4922e-01, 9.6122e-01, 4.6202e-01, 2.3005e-01, 4.8874e-01,
             4.0460e-01, 6.2650e-01, 5.8765e-01, -6.0967e-01, -3.3920e-01,
             7.1510e-01, -4.3277e-02],
            [3.2249e-01, 8.2594e-01, 6.5295e-01, 7.3791e-01, 7.6101e-01,
             4.2735e-01, 6.5183e-01, 6.4362e-01, 1.5938e-01, -2.9170e-01,
             3.1579e-01, -8.8870e-01],
            [6.6968e-01, 9.0331e-02, 2.8199e-01, 9.2455e-01, 1.7525e-01,
             4.7853e-01, 4.8160e-01, 5.7773e-01, -3.0864e-01, 5.1476e-01,
             -7.9101e-01, -1.1861e-01],
            [2.7732e-02, 1.0841e-01, 2.1101e-01, 6.2058e-01, 9.8049e-01,
             6.4204e-01, 3.4959e-01, 5.0169e-01, -6.5881e-01, 2.6454e-01,
             3.8569e-01, -5.8926e-01]], device='cuda:0')
    """
    pred_labels = (torch.tensor([[0.3372, 0.5157, 0.5284, 0.4635, 0.5544, 0.4327, 0.3189, 0.5812],
             [0.5781, 0.4905, 0.5339, 0.4268, 0.4897, 0.6308, 0.6414, 0.4740],
             [0.2061, 0.3661, 0.1458, 0.6121, 0.5929, 0.5438, 0.4416, 0.3271],
             [0.2700, 0.4052, 0.4693, 0.5098, 0.4908, 0.4758, 0.5668, 0.5164],
             [0.6365, 0.4771, 0.4589, 0.7046, 0.5595, 0.5780, 0.5661, 0.4998],
             [0.2754, 0.5091, 0.4049, 0.5810, 0.7124, 0.6788, 0.6100, 0.3619],
             [0.6637, 0.5349, 0.5945, 0.5632, 0.6545, 0.5295, 0.5748, 0.5498],
             [0.5326, 0.7467, 0.5204, 0.5630, 0.7092, 0.3208, 0.5279, 0.4830]],
            device='cuda:0'),
     torch.tensor([[0.7812, 0.3851, 0.4784, -0.1118],
              [0.8436, -0.1965, 0.1472, -0.4775],
              [0.6984, -0.0481, 0.6981, -0.1505],
              [0.8615, -0.0768, 0.3727, -0.3362],
              [0.6646, -0.2273, -0.1368, -0.6985],
              [0.4675, -0.2671, -0.0574, -0.8407],
              [0.6767, -0.1767, -0.3040, -0.6469],
              [0.4652, 0.0150, 0.8848, 0.0230]], device='cuda:0'))
    
    pred_labels = torch.autograd.Variable(torch.cat(pred_labels, dim=-1), requires_grad=True)

    true_labels = (torch.tensor([[0.2898, 0.1687, 0.9747, 0.9154, 0.4006, 0.4486, 0.3756, 0.5870],
                                 [0.5626, 0.9912, 0.0508, 0.7745, 0.1807, 0.6210, 0.6083, 0.6057],
                                 [0.2327, 0.4654, 0.1832, 0.9048, 0.6087, 0.5368, 0.4564, 0.3573],
                                 [0.1815, 0.5594, 0.6331, 0.3633, 0.2536, 0.4522, 0.5790, 0.6382],
                                 [0.3712, 0.5009, 0.7068, 0.6331, 0.9695, 0.5820, 0.5766, 0.5710],
                                 [0.2911, 0.9183, 0.0563, 0.9679, 0.6560, 0.6426, 0.6298, 0.4838],
                                 [0.5410, 0.6685, 0.3506, 0.3073, 0.2177, 0.5277, 0.6017, 0.6157],
                                 [0.9316, 0.8704, 0.5318, 0.9672, 0.6645, 0.3505, 0.5409, 0.5840]],device='cuda:0'),
                   torch.tensor([[-0.5095, -0.7611, 0.2858, -0.2818],
                                  [0.4112, 0.6584, -0.6029, 0.1844],
                                  [-0.1513, -0.8056, 0.5247, 0.2298],
                                  [0.5389, 0.5338, -0.5739, -0.3086],
                                  [-0.2268, 0.6403, -0.4713, -0.5625],
                                  [0.2173, 0.4509, 0.0581, 0.8638],
                                  [-0.2169, 0.0588, 0.5052, 0.8332],
                                  [-0.7467, -0.3480, 0.2128, -0.5254]], device='cuda:0'))
    true_labels = torch.cat(true_labels, dim=-1)
    """
    #true_p = torch.tensor([[0.0084, 0.8, 0.0191, 0.8786, 0.1, 0.3, 0.8, 0.8, 0, 0, 1, 1]],
    #                      device='cuda:0', dtype=torch.float64)
    #pred_p = torch.tensor([[0.0084, 0.8, 0.0191, 0.8786, 0.1, 0.3, 0.8, 0.8, 0, 0, 0, 1]],
    #   device='cuda:0', dtype=torch.float64)

    #pred_p = torch.tensor([[0.3633, 0.4566, 0.4327, 0.1267, 0.1300, 0.5209, 0.3704, 0.5967]], dtype=torch.float32, requires_grad=req_grad).to(device)
    #true_p = torch.tensor([[0.3800, 0.4600, 0.4200, 0.1202, 0.1064, 0.5155, 0.3829, 0.5830]], dtype=torch.float32, requires_grad=req_grad).to(device)
    #print(torch.autograd.gradcheck(loss.__call__, (true_p, pred_p)))
    loss = ChamferLoss(32, device, reduce=True)

    #loss_chamfer(, true_labels)

    l = loss(true, pred)
    #l.backward()
    print(l)
    print("-----------")
    #print(pred.grad)

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
