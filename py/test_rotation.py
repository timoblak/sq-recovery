from keras.models import *
from data_generators import *
from utils import *
from models import quaternion_loss, quaternion_loss_np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    abs_errs = []
    m = load_model("../models/cnn_100k_2.h5",
                   custom_objects={"quaternion_loss":quaternion_loss})
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
        params_pred = m.predict(img[None, :, :, None])
        block, quat = params_pred

        # quat /= np.sqrt( (quat ** 2).sum())

        # Matrix or quaternion
        M = quat2mat(quat.ravel())
        # M = raveled_mat.reshape((3, 3))

        params_pred = np.concatenate([block.ravel(), M.ravel(), quat.ravel()])
        # preds = np.concatenate([block.ravel(), M.ravel(), mat2quat(M)])

        for i in [0, 1, 2]:
            params_pred[i] = 50 * params_pred[i] + 25
        for i in [5, 6, 7]:
            params_pred[i] = 255 * params_pred[i]

        # print("-" * 80)
        # print( ("%+9.4f " * 21) % tuple(params_true))
        # print( ("%+9.4f " * 21) % tuple(preds))
        #print(quat, q)

        print("\tBlock [true - false]")
        print(params_true[-4:])
        print(params_pred[-4:])
        print("\tRotation [true - false]")
        print(params_true[0:8])
        print(params_pred[0:8])
        print("\tDiff [quat_distance - abs]")
        print(quaternion_loss_np(q, quat.ravel()))
        print(np.abs(params_true[0:8] - params_pred[0:8]), np.sum(np.abs(params_true[0:8] - params_pred[0:8])))

        #print(params_true)

        if debug:
            command = get_command(scanner_location, "tmp2.bmp", params_pred[:-4])
            os.system(command)
            img_hat = cv2.imread("tmp2.bmp").mean(2) / 255
            img[:, -1] = 1
            img_hat[:, 0] = 1
            cv2.imshow("image", np.concatenate((img, img_hat), axis=1))
            cv2.waitKey(0)

        abs_errs.append(np.abs(params_true - params_pred))

    mae = sum(abs_errs) / N
    print(("%+9.4f " * 21) % tuple(mae))
