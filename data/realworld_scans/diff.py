import cv2
import numpy as np

img_n = "1"

file1 = "realworld/" + img_n + "p.bmp"
file2 = "realworld/" + img_n + ".bmp"

img1 = cv2.imread(file1, 0)
img2 = cv2.imread(file2, 0)

cv2.imshow("diff1", img1)
cv2.imshow("diff2", img2)
cv2.waitKey(0)

img = cv2.absdiff(img1, img2)
print(np.sum(img))

cv2.imshow("diff", img)
cv2.imwrite("realworld/" + img_n + "_diff.bmp", img)
cv2.waitKey(0)