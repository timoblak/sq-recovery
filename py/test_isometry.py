from keras.models import *
from keras.callbacks import *
from data_generators import *
import keras.backend as K
from utils import *
from models import get_model
import shutil
import time


if __name__ == "__main__":

    model_path = "../models/cnn_isometry_100k.h5"
    scanner_location = "../"

    whole_annot = ""
    abs_errs = []
    m = load_model(model_path)
    N = 20000
    debug = True
    pos = np.array([128, 128, 128], dtype="float64")
    times = []
    for num_img in tqdm(range(N)):
        im_name = "{:05d}".format(num_img)
        dims = np.random.uniform(25, 76, (3,))
        shape = np.random.uniform(0.01, 1.0, (2,))
        pos_new = pos + np.random.uniform(-40, 41, (3,))
        q = q = np.array([1, 1, 1, 0], dtype="float64")
        M = quat2mat(q)
        params = np.concatenate((dims.astype("float64"), shape, pos_new.ravel().astype("float64"), M.ravel()))
        command = get_command(scanner_location, "tmp.bmp", params)
        print(command)
        os.system(command)

        params_true = np.concatenate([dims.astype("float64"), shape, pos_new.ravel().astype("float64")])

        img = cv2.imread("tmp.bmp").mean(2)

        img /= 255

        t0 = time.time()
        preds = m.predict(img[None, :, :, None])
        if num_img > 0:
            times.append(time.time() - t0)
        #print(times)
            print(np.sum(times)/len(times))

        #print(preds)
        preds = preds[0]

        for i in [0, 1, 2]:
            preds[i] = 50 * preds[i] + 25
        for i in [5, 6, 7]:
            preds[i] = 255 * preds[i]

        if debug:
            show_preds = np.concatenate([preds, M.ravel()])
            command = get_command(scanner_location, "tmp2.bmp", show_preds)
            print(command)
            os.system(command)
            img_hat = cv2.imread("tmp2.bmp").mean(2) / 255
            img[:, -1] = 1
            img_hat[:, 0] = 1
            cv2.imshow("image", np.concatenate((img, img_hat), axis=1))
            cv2.waitKey(0)

    mae = sum(abs_errs) / N
    print(("%+9.4f " * 8) % tuple(mae))
