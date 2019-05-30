import tqdm
import numpy as np
import os 
from utils import quat2mat, randquat


if __name__ == "__main__":
    a1, a2, a3 = np.random.uniform(25, 75, (3,))
    e1, e2     = np.random.uniform(0.05, 1.0, (2,))
    pos = np.array([128.0, 128.0, 128.0])


    dst = os.path.expanduser("~/superblocks/data_val/")
    n_data = 1000

    f1 = open("gen_rand_rot_val.sh", "w")
    f2 = open("data_rot_val.csv", "w")

    for i in tqdm(range(n_data)):

        fn = os.path.join(dst, "%06d.bmp" % i)
        command = "./scanner " + fn + " "
        dims = np.random.uniform(25, 76, (3,))
        command += ("%f " * 3) % tuple(dims)
        shape = np.random.uniform(0.01, 1.0, (2,))
        command += ("%f " * 2) % tuple(shape)
        pos_new = pos + np.random.uniform(-40, 41, (3,))
        command += "%f %f %f " % tuple(pos_new.ravel())

        q = randquat()
        M = quat2mat(q)

        command += ("%f " * 9) % tuple(M.ravel())
        command = command[:-1] + "\n"
        f1.write(command)

        #.csv file for loading parameter values into data generator
        params = np.concatenate([dims, shape, pos_new, M.ravel(), q])
        params = params.astype("float64").ravel()
        csvline = (fn + "," + ("%f," * 21) % tuple(params))[:-1] + "\n"
        f2.write(csvline)
    f1.close()
    f2.close()
    sys.exit(0)
