import cv2
import os


img_folder = "imgs/"
img_predict_folder = "imgs_predict/"

for fname in os.listdir(img_folder):

    im = cv2.imread(img_folder + fname, 0)

    height, width = im.shape

    p = (width - height)//2

    im = im[:, p:width-p]
    im = cv2.resize(im, (256, 256))

    cv2.imshow("asd", im)
    cv2.imwrite(img_predict_folder + fname.replace(".png", "_norma.png"), im)


