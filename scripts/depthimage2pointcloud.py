import cv2
import os
import numpy as np

for img_name in os.listdir("classic/"):
    if img_name.split(".")[1] != "bmp":
        continue

    img = cv2.imread("classic/"+img_name, 0)
    fimg = cv2.flip(img, 0)

    nz = np.nonzero(fimg)

    ls = np.transpose((nz[1], nz[0], fimg[np.nonzero(fimg)]))

    text = ""
    for p in ls:
        text += str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"


    with open("classic/" + img_name.replace(".bmp", "_points") + ".txt", "w") as handle:
        handle.write(text)
    """
    #cv2.imshow("asd", img)
    #cv2.imshow("asd1", fimg)
    #cv2.waitKey(0)
    """