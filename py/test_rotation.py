from keras.models import *
from data_generators import *
from utils import randquat, quat2mat


if __name__ == "__main__":

    abs_errs = []
    m = load_model("models/klemen_baseline_cnn_quats_400k.h5")
    scanner_location = "../"

    N = 20000
    debug = True
    pos = np.array([128, 128, 128], dtype="float64")
    for _ in tqdm(range(N)):
        dims = np.random.randint(25, 76, (3,))
        shape = np.random.uniform(0.01, 1.0, (2,))
        pos_new = pos + np.random.uniform(-40, 41, (3,))
        q = randquat()
        M = quat2mat(q)
        params = np.concatenate((dims.astype("float64"), shape, pos_new.ravel().astype("float64"), M.ravel()))
        command = get_command(scanner_location, "tmp.bmp", params)
        os.system(command)

        params_true = np.concatenate([params, q])

        img = cv2.imread("tmp.bmp").mean(2) / 255
        preds = m.predict(img[None, :, :, None])
        block, quat = preds

        # quat /= np.sqrt( (quat ** 2).sum())

        # Matrix or quaternion
        M = quat2mat(quat.ravel())
        # M = raveled_mat.reshape((3, 3))

        preds = np.concatenate([block.ravel(), M.ravel(), quat.ravel()])
        # preds = np.concatenate([block.ravel(), M.ravel(), mat2quat(M)])

        for i in [0, 1, 2]:
            preds[i] = 50 * preds[i] + 25
        for i in [5, 6, 7]:
            preds[i] = 255 * preds[i]

        # print("-" * 80)
        # print( ("%+9.4f " * 21) % tuple(params_true))
        # print( ("%+9.4f " * 21) % tuple(preds))
        print(quat, q)
        print(preds)
        print(params_true)

        if debug:
            command = get_command(scanner_location, "tmp2.bmp", preds[:-4])
            os.system(command)
            img_hat = cv2.imread("tmp2.bmp").mean(2) / 255
            img[:, -1] = 1
            img_hat[:, 0] = 1
            cv2.imshow("image", np.concatenate((img, img_hat), axis=1))
            plt.show()

        abs_errs.append(np.abs(params_true - preds))

    mae = sum(abs_errs) / N
    print(("%+9.4f " * 21) % tuple(mae))
