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
    image_location = "../data/example_imgs/000000.bmp"

    TRAINED_MODEL = "trained_models/model_explicit.pt"

    net = ResNetSQ(outputs=4, pretrained=False).to(device)

    # Load model weights and put network in to EVAL mode
    epoch, net, _, _ = load_model(TRAINED_MODEL, net, None)
    net.eval()

    img1 = cv2.imread(image_location, 0)

    img_np = np.expand_dims(np.expand_dims(img1.astype(np.float32), 0), 0)
    img_np = img_np/255

    data = torch.from_numpy(img_np).to(device)

    pred_labels = net(data)
    a = pred_labels[0].detach().cpu().numpy()
    e = pred_labels[1].detach().cpu().numpy()
    t = pred_labels[2].detach().cpu().numpy()
    q = pred_labels[3].detach().cpu().numpy()

    print("Predicted parameters: ")
    print("Size a:", a*255)
    print("Shape e:", e)
    print("Position t:", t*255)
    print("Rotation q:", q)

    cv2.imshow("image", img1)
    cv2.waitKey(0)
