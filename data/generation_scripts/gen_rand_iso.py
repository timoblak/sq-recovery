from tqdm import tqdm
import numpy as np
import os
from utils import quat2mat


a1, a2, a3 = np.random.uniform(25, 75, (3,))
e1, e2     = np.random.uniform(0.1, 1.0, (2,))
pos = np.array([128.0, 128.0, 128.0])

q = np.array([1, 1, 1, 0], dtype="float64")

M = quat2mat(q)
n_data = 50000

dst = "data_iso2"

f1 = open("gen_rand_iso2.sh", "w")
f2 = open("data_iso2.csv", "w")

for i in tqdm(range(n_data)):
    fn = os.path.join(dst, "%06d.bmp" % i)
    command = "./scanner " + fn + " "
    dims = np.random.uniform(25, 75, (3,))
    command += ("%d " * 3) % tuple(dims)
    shape = np.random.uniform(0.1, 1.0, (2,))
    command += ("%f " * 2) % tuple(shape)
    pos_new = pos + np.random.uniform(-40, 40, (3,))
    command += "%f %f %f " % tuple(pos_new.ravel())
    command += ("%f " * 9) % tuple(M.ravel())
    command += "\n"
    f1.write(command)

    # .csv file for loading parameter values into data generator
    params = np.concatenate([dims, shape, pos_new, M.ravel(), q])
    params = params.astype("float64").ravel()
    csvline = (fn + "," + ("%f," * 21) % tuple(params))[:-1] + "\n"
    f2.write(csvline)

f1.close()
f2.close()
