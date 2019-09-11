import os
from tqdm import tqdm
import cv2
import numpy as np


def visualize_batch(X, Y1, Y2, selected, mode=""):
    for i in range(len(X)):
        cv2.imshow("ad", X[i])
        print(Y1[i], Y2[i], selected[i])
        cv2.waitKey(0)


def parse_csv(csvfile):
    with open(csvfile, "r") as f:
        s = f.read()
    lines = s.split("\n")[:-1]
    labels = []
    print("Parsing csv...")
    for line in tqdm(lines):
        split_line = line.split(",")
        pl2 = [split_line[0]]
        for i in range(1, 9):
            pl2.append(float(split_line[i]))

        # Normalize values
        for i in [1, 2, 3]:
            pl2[i] = (pl2[i] - 25) / 50
        for i in [6, 7, 8]:
            pl2[i] /= 255.0

        for i in range(-4, 0):
           pl2.append(float(split_line[i]))
        labels.append(pl2)
    return np.array(labels)


def load_dataset(dataset_location):
    np_archive = dataset_location + "/data_archive.npy"
    if not os.path.isfile(np_archive):
        image_files = []

        for fl in os.listdir(dataset_location):
            # Load only bmp files
            if fl.endswith(".bmp"):
                image_files.append(fl)
        image_files = sorted(image_files)

        dataset = np.zeros((len(image_files), 256, 256, 1), dtype="uint8")
        for i in tqdm(range(len(image_files))):
            path = os.path.join(dataset_location, image_files[i])
            im = cv2.imread(path)[:, :, 0].astype("float32")
            dataset[i, :, :, 0] = im
        np.save(np_archive, dataset)
    else:
        print("Loading existing... (" + dataset_location + ")")
        dataset = np.load(np_archive)
    return dataset


def data_gen_iso(labels, img_dir, batch_size):
    dataset = load_dataset(img_dir)

    while True:
        p = np.random.permutation(len(dataset))
        for i in range(0, dataset.shape[0], batch_size):
            selected_samples = p[i:i + batch_size]
            X_batch = dataset[selected_samples] / 255
            Y_batch = np.array(labels[selected_samples, 1:9], dtype=np.float32)
            yield X_batch, Y_batch


def data_gen_quats(labels, img_dir, batch_size):
    dataset = load_dataset(img_dir)

    while True:
        p = np.random.permutation(len(dataset))
        for i in range(0, dataset.shape[0], batch_size):
            selected_samples = p[i:i+batch_size]
            X_batch = dataset[selected_samples] / 255
            Y1_batch = labels[selected_samples, 1:9]
            Y2_batch = labels[selected_samples, -4:]
            Y_batch = np.concatenate([Y1_batch, Y2_batch], axis=-1).astype(np.float32)
            yield X_batch, Y_batch

