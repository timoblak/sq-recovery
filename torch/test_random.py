import cv2
import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
from classes import IoUAccuracy, ExplicitLoss, ImplicitLoss
from helpers import quat2mat, get_command
from quaternion import conjugate, randquat, multiply, to_magnitude
from helpers import load_model
from models import ResNetSQ


if __name__ == "__main__":
    
    device = "cuda:0"
    scanner_location = "../data/"

    TRAINED_MODEL = "trained_models/model_explicit.pt"
    N = 1000
    DEBUG = True
    f = open("results.txt", "a")

    net = ResNetSQ(outputs=4, pretrained=False).to(device)
    acc = IoUAccuracy(render_size=128, device=device, full=True)

    # Load model weights and put network in to EVAL mode
    epoch, net, _, _ = load_model(TRAINED_MODEL, net, None)
    net.eval()

    accs = []
    for i in tqdm(range(N)):
        # Create a random instance 
        a = np.random.uniform(25, 75, (3,)).astype("float64")
        e = np.random.uniform(0.1, 1.0, (2,)).astype("float64")
        t = np.array([128, 128, 128], dtype="float64") + np.random.uniform(-40, 40, (3,))
        q = randquat()
        M1 = quat2mat(q)

        # Prepare parameters (non-normalized) and create image
        params = np.concatenate((a, e, t, M1.ravel()))
        command = get_command(scanner_location, "tmp1.bmp", params)
        os.system(command)

        # Read image and predict on it 
        img1 = cv2.imread("tmp1.bmp", 0)
        img_np = np.expand_dims(np.expand_dims(img1.astype(np.float32), 0), 0)
        img_np = img_np/255 #/255

        data = torch.from_numpy(img_np).to(device)

        pred_labels = net(data)

        # Create parameters to evaluate 
        block_true = torch.tensor(np.expand_dims(np.concatenate((a/255, e, t/255)), 0), dtype=torch.double, device=device)
        quat_true = torch.tensor(np.expand_dims(q, axis=0), dtype=torch.double, device=device)
        
        true_param = torch.cat((block_true, quat_true), dim=-1)
        
        print("---------- Example "+str(i)+" ----------", file=f)
        print("True params:", a, e, t, q, file=f)
        ap, ep, tp, qp = pred_labels
        ap = ap[0].detach().cpu().numpy()*255
        ep = ep[0].detach().cpu().numpy()
        tp = tp[0].detach().cpu().numpy()*255
        qp = qp[0].detach().cpu().numpy()
        print("Pred params:", ap, ep, tp, qp, file=f)
        
        #print("--------------")
        print(ap, ep, tp, qp)
        M2 = quat2mat(qp)
        params2 = np.concatenate((ap, ep, tp, M2.ravel()))
        command = get_command(scanner_location, "tmp3.bmp", params2)
        os.system(command)

        img3 = cv2.imread("tmp3.bmp", 0)
        image = np.hstack([img1, img3])

        accuracy = acc(true_param.double(), torch.cat(pred_labels, dim=1).double())

        acc_np = accuracy.detach().cpu().numpy()

        print("- Accuracy:", acc_np*100, file=f)
        accs.append(acc_np)
        
    accs = np.array(accs)
    a = np.array(accs[:, 0])
    a_full = np.array(accs[:, 1])
    m = np.array(accs[:, 2])
    print("--Rot::")
    print("Mean: ", a.mean()) 
    print("Std: ", a.std())      
    print("--Full::")
    print("Mean: ", a_full.mean()) 
    print("Std: ", a_full.std())

    with open("accs_angle1.pkl", "wb") as handle: 
        pickle.dump([a,m], handle)
