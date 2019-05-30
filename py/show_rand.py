from startup import *

dst = "data_iso"

fns = np.random.permutation(os.listdir(dst))[:72]
fns = [os.path.join(dst, fn) for fn in fns]

imgs = np.array([cv2.imread(fn)[:, :, ::-1] for fn in fns])
print(imgs.shape, imgs.dtype, imgs.min(), imgs.max())
tensorshow(imgs, shape=(12, 6), size=(37.5, 19))
sys.exit(0)

gy, gx = np.gradient(imgs.astype("float64"), axis=(1, 2))
g_imgs = np.sqrt(gy**2 + gx**2)
g_imgs -= g_imgs.min()
g_imgs /= g_imgs.max()
g_imgs *= 255
disp = np.concatenate((imgs.astype("float64")[:25], g_imgs[:25]), axis=2)
tensorshow(disp, size=(37.5, 19))

sys.exit(0)

imgs = np.mean(imgs, axis=3, dtype="float64")
imgs -= imgs.mean()
imgs /= imgs.std()

fn1 = os.path.join("data1_png", np.random.choice(os.listdir("data1_png")))
fn2 = os.path.join("data2_png", np.random.choice(os.listdir("data2_png")))
im1 = cv2.imread(fn1)
im2 = cv2.imread(fn2)

plt.figure(0, (36, 19))
plt.subplot(121)
plt.imshow(im1, interpolation="nearest")
hideAxes()
plt.subplot(122)
plt.imshow(im2, interpolation="nearest")
hideAxes()
plt.tight_layout()
plt.show()
