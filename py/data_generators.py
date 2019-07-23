import os
from tqdm import tqdm
import cv2
import numpy as np


def parse_csv_iso(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")[:-1]
    parsed_lines = []
    for line in tqdm(lines):
        pl = line.split(",")
        pl2 = [pl[0]]
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            pl2.append(float(pl[i]))
        for i in [1, 2, 3]:
            pl2[i] = (pl2[i] - 25) / 50
        for i in [6, 7, 8]:
            pl2[i] /= 255.0
        parsed_lines.append(pl2)
    return parsed_lines


def data_gen_iso(csvfile, img_dir, batch_size, debug=False, mode=""):
    archive = os.path.expanduser("~/superblocks/data_iso" + mode + "/data_iso.npy")
    parsed_lines = parse_csv_iso(csvfile)
    fns = sorted(os.listdir(img_dir))
    if not os.path.isfile(archive):
        all_imgs = np.zeros((len(fns), 256, 256, 1), dtype="uint8")
        for i in tqdm(range(len(fns))):
            path = os.path.join(img_dir, fns[i])
            im = cv2.imread(path)[:, :, 0].astype("float32")
            all_imgs[i, :, :, 0] = im
        np.save(archive, all_imgs)
    else:
        all_imgs = np.load(archive)

    X_batch = np.zeros((batch_size, 256, 256, 1), dtype="float32")
    Y_batch = np.zeros((batch_size, 8), dtype="float32")
    while True:
        for batch_ind in range(batch_size):
            im_ind = np.random.randint(len(parsed_lines))
            line = parsed_lines[im_ind]
            if debug:
                print(line, fns[im_ind])
            img = all_imgs[im_ind].mean(2).astype("float32") / 255
            X_batch[batch_ind, :, :, 0] = img.astype("float32")
            Y_batch[batch_ind] = line[1:9]
        yield X_batch, Y_batch

def visualize_batch(X, Y1, Y2, selected, mode=""):
    for i in range(len(X)):
        cv2.imshow("ad", X[i])
        print(Y1[i], Y2[i], selected[i])
        cv2.waitKey(0)


def data_gen_quats(csvfile, img_dir, batch_size, debug=False, mode=""):
    archive = "../data/data" + mode + "/data_rot.npy"
    labels = csvfile
    fns = sorted(os.listdir(img_dir))

    if not os.path.isfile(archive):
        all_imgs = np.zeros((len(fns), 256, 256, 1), dtype="uint8")
        for i in tqdm(range(len(fns))):
            path = os.path.join(img_dir, fns[i])
            #priint(path)
            im = cv2.imread(path)[:, :, 0].astype("float32")
            all_imgs[i, :, :, 0] = im
        np.save(archive, all_imgs)
    else:
        print("Loading existing... (" +mode+ ")")
        all_imgs = np.load(archive)
    print(all_imgs.shape, labels.shape, mode)

    while True:
        p = np.random.permutation(len(all_imgs))
        for i in range(0, all_imgs.shape[0], batch_size):
            #im_ind = np.random.randint(len(labels))
            selected = p[i:i+batch_size]
            X_batch = all_imgs[selected] / 255
            Y1_batch = labels[selected, 1:9]
            Y2_batch = labels[selected, -4:]
            if mode=="_rot_val":
                visualize_batch(X_batch, Y1_batch, Y2_batch, selected)
            #print(X_batch.shape, Y1_batch.shape, Y2_batch.shape)
            yield X_batch, [Y1_batch, Y2_batch]

    #while True:
    #    for batch_ind in range(batch_size):
    #        im_ind = np.random.randint(len(labels))
    #        line = labels[im_ind]
    #        if debug:
    #            print(line, fns[im_ind])
    #        img = all_imgs[im_ind].mean(2).astype("float32") / 255
    #        X_batch[batch_ind, :, :, 0] = img.astype("float32")
    #        Y1_batch[batch_ind] = line[1:9]
    #        Y2_batch[batch_ind] = line[-4:]
    #    yield X_batch, [Y1_batch, Y2_batch]


def parse_csv_full(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")[:-1]
    parsed_lines = []
    for line in tqdm(lines):

        pl = line.split(",")
        pl2 = [pl[0]]
        for i in range(1, 18):
            pl2.append(float(pl[i]))
        for i in [1, 2, 3]:
            pl2[i] = (pl2[i] - 25) / 50
        for i in [6, 7, 8]:
            pl2[i] /= 255.0
        parsed_lines.append(pl2)
    return parsed_lines


def data_gen_full(csvfile, img_dir, batch_size, debug=False):
    parsed_lines = parse_csv_full(csvfile)
    fns = sorted(os.listdir(img_dir))
    if not os.path.isfile("data_full.npy"):
        all_imgs = np.zeros((len(fns), 256, 256, 1), dtype="uint8")
        for i in tqdm(range(len(fns))):
            path = os.path.join(img_dir, fns[i])
            im = cv2.imread(path)[:, :, 0].astype("float32")
            all_imgs[i, :, :, 0] = im
        np.save("data_full.npy", all_imgs)
    else:
        all_imgs = np.load("data_full.npy")

    X_batch = np.zeros((batch_size, 256, 256, 1), dtype="float32")
    Y1_batch = np.zeros((batch_size, 8), dtype="float32")
    Y2_batch = np.zeros((batch_size, 9), dtype="float32")
    img = None
    while True:
        fns = []
        for batch_ind in range(batch_size):
            im_ind = np.random.randint(80000) #len(parsed_lines))
            line = parsed_lines[im_ind]
            if debug:
                fns.append(line[0])
            img = all_imgs[im_ind].mean(2).astype("float32") / 255
            X_batch[batch_ind, :, :, 0] = img.astype("float32")
            Y1_batch[batch_ind] = line[1 : 9]
            Y2_batch[batch_ind] = line[9:]
        if debug:
            yield X_batch, [Y1_batch, Y2_batch, fns]
        else:
            yield X_batch, [Y1_batch, Y2_batch]


def parse_csv_quats(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")[:-1]
    parsed_lines = []
    print("Parsing csv...")
    for line in tqdm(lines):
        pl = line.split(",")
        pl2 = [pl[0]]
        for i in range(1, 9):
            pl2.append(float(pl[i]))
        for i in [1, 2, 3]:
            pl2[i] = (pl2[i] - 25) / 50
        for i in [6, 7, 8]:
            pl2[i] /= 255.0
        for i in range(-4, 0):
           pl2.append(float(pl[i]))
        parsed_lines.append(pl2)
    return np.array(parsed_lines)





if __name__ == "__main__":
    trG = data_gen_full("data_full.csv", "/opt/data/superkvadriki/data_full", 
                        batch_size=128, debug=True)
    for _ in range(4):
        X, (Y1, Y2, fns) = next(trG)
        command = "./scanner tmp.bmp "

        for i in [0, 1, 2]:
            Y1[:, i] = (50 * Y1[:, i]) + 25
        for i in [5, 6, 7]:
            Y1[:, i] *= 255.0

        for n in Y1[88, :]:
            command += "%f " % n
        for n in Y2[88, :]:
            command += "%f " % n

        os.system(command)

        print(fns[88])

        ref = cv2.imread(fns[88]).mean(2)
        real= cv2.imread("tmp.bmp").mean(2)
        print(np.abs(ref - real).mean())
        imshow(np.concatenate((ref, real), axis=1))
        print(X.shape, Y1.shape, Y2.shape)
        print(np.abs(Y2 - Y2.mean()).mean())
#        tensorshow(X, shape=(16, 8), size=(37.5, 19))
