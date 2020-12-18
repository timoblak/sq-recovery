import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from time import time, sleep
from quaternion import rotate, mat_from_quaternion_np, conjugate_np, multiply_np, to_magnitude_np, conjugate, multiply, to_magnitude
from helpers import randquat, slerp, quat2mat, get_command
from classes import ExplicitLoss, ImplicitLoss, IoUAccuracy, LeastSquares
import torch
from sklearn.preprocessing import normalize
import cv2
import os


def sample_points_uniform(parameters, rad_resolution=0.5):
    a1, a2, a3 = parameters[:3]
    e1, e2 = parameters[3:5]

    theta, gamma = np.arange(-np.pi, np.pi, rad_resolution), np.arange(-np.pi / 2, np.pi / 2, rad_resolution)
    w, h = len(theta), len(gamma)

    pts = np.zeros(shape=(h * w, 3))
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    iter = 0
    for t in theta:
        pts[iter * h:(iter + 1) * h, 0] = a1 * np.sign(cg * np.cos(t)) * (np.abs(cg) ** e1) * (np.abs(np.cos(t)) ** e2)
        pts[iter * h:(iter + 1) * h, 1] = a2 * np.sign(cg * np.sin(t)) * (np.abs(cg) ** e1) * (np.abs(np.sin(t)) ** e2)
        pts[iter * h:(iter + 1) * h, 2] = a3 * np.sign(sg) * (np.abs(sg) ** e1)
        iter += 1
    return pts


def quaternion_diffs(q):
    quat_diffs = []
    quat_diffs.append(q[0])
    for qi in range(1, len(q)):

        app = multiply_np(q[qi], conjugate_np(q[qi-1]))
        quat_diffs.append(app)

    quat_diffs.append(conjugate_np(q[-1]))
    return np.stack(quat_diffs)


def init_visualization():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visu", width=800, height=600)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    return vis


def randsq():
    return np.concatenate((np.random.uniform(0.1, 0.3, (3,)), np.random.uniform(0.1, 1, (2,)), np.random.uniform(0.34, 0.65, (3,))))


if __name__ == "__main__":
    device = torch.device("cuda:0")
    req_grad = False

    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (800, 600))

    granularity = 32

    sharpness = 260
    tau = 1.5

    #loss = ImplicitLoss(render_size=64, device=device, tau=tau, sigmoid_sharpness=sharpness)
    loss = ExplicitLoss(render_size=granularity, device=device)
    acc = IoUAccuracy(render_size=128, device=device, full=True)

    torch.autograd.set_detect_anomaly(True)

    true_parameters = np.array([0.17840092, 0.29169756, 0.19272356, 0.564326, 0.850042, 0.5160052,0.51887995, 0.41229093, 0.468217, 0.567843, -0.355409, 0.576204])
    pred_parameters = np.concatenate([randsq(), randquat()])

    q_true = true_parameters[8:]
    q_pred = pred_parameters[8:]

    # Create SQ pointcloud
    params_true = true_parameters[:8]
    params_pred = pred_parameters[:8]

    M = quat2mat(q_true[-4:])
    params = np.concatenate((params_true[:3] * 255., params_true[3:5], params_true[5:8] * 255, M.ravel()))
    command = get_command("../", "visu_true.bmp", params)
    os.system(command)

    img = cv2.imread("visu_true.bmp", 0)/255
    true_img = torch.from_numpy(img).to(device)
    true_img = true_img.unsqueeze(0).unsqueeze(0)

    pts = sample_points_uniform(params_true, rad_resolution=0.1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Create convex hull
    hull, _ = pcd.compute_convex_hull()
    hull.compute_vertex_normals()

    # Line mesh for true quadric
    hull_ls1 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls1.paint_uniform_color((1, 0, 0))

    mat_transform_true = np.eye(4)
    mat_transform_true[:3, :3] = mat_from_quaternion_np(q_true)[0]
    mat_transform_true[:3, 3] = params_true[5:8]

    hull_ls1.transform(mat_transform_true)

    debug = False
    if not debug:
        vis = init_visualization()
        vis.add_geometry(hull_ls1)
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2020-08-31-14-29-43.json")
        #ctr.convert_from_pinhole_camera_parameters(parameters)

    lr = 0.001
    i = 0
    while True:
        #ctr.convert_from_pinhole_camera_parameters(parameters)
        pts = sample_points_uniform(params_pred, rad_resolution=0.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Create convex hull
        hull_pred, _ = pcd.compute_convex_hull()
        hull_pred.compute_vertex_normals()
        if not debug:
            vis.add_geometry(hull_pred, reset_bounding_box=False)

        mat_q = mat_from_quaternion_np(q_pred)[0]

        #hull_pred.rotate(mat_q, [0, 0, 0])
        mat_transform = np.eye(4)
        mat_transform[:3, :3] = mat_q
        mat_transform[:3, 3] = params_pred[5:8]

        hull_pred.transform(mat_transform)
        true = torch.tensor(
            [
                np.concatenate([params_true, q_true])
            ], device='cuda:0')
            

        pred = torch.tensor(
            [
                #q_pred
                np.concatenate([params_pred, q_pred])
            ],
            device='cuda:0', requires_grad=True)

        #l = l_a + l_e + l_t + l_q

        l = loss(true, pred)
        #a = acc(true, pred[:, 8:])
        a = acc(true, pred)

        diff = to_magnitude(multiply(torch.from_numpy(q_true), conjugate(torch.from_numpy(q_pred))))
        loss_np = l.detach().cpu().item()
        acc_np = a.detach().cpu().item()

        l.backward()

        print("Iter", i)
        print("Predicted:", q_pred)
        print("Predicted:", normalize([q_pred]))
        print("True", q_true)
        print("Grads:", pred.grad)
        print("Loss:", loss_np)
        print("Accuracy:", acc_np)
        print("Angle difference:", diff)
        print("Lr:", lr)
        print("---------------------------------------")

        #if i > 3500:
        #    lr = 0.005
        #if i > 10000:
        #    lr = 0.001
        gradient = pred.grad.detach().cpu().numpy()[0]
        #exit()
        q_pred -= lr * gradient[8:]
        q_pred = normalize([q_pred])[0]

        params_pred -= lr * gradient[:8]

        if not debug:
            vis.update_geometry(hull_pred)
            vis.poll_events()
            vis.update_renderer()

        #if i < 100 and i % 10 == 0 or i % 100 == 0 :
        #    image = vis.capture_screen_float_buffer(False)
        #    plt.imsave("iteration_images/render_"+str(i)+".png", np.asarray(image), dpi=1)
        #img = np.asarray(vis.capture_screen_float_buffer(True))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = (img * 255).astype('uint8')
        #out.write(img)
        #cv2.imshow("asdasd", img)
        #cv2.waitKey()


        sleep(0.0)
        #hull_pred.rotate(mat_q.T, [0, 0, 0])
        i+=1
        if not debug:
            vis.remove_geometry(hull_pred, reset_bounding_box=False)


    out.release()
    vis.destroy_window()

"""

tensor([[ 0.1166,  0.1219,  0.1367,  0.7549,  0.5152,  0.5910,  0.3958,  0.6315,
          0.4587, -0.8290,  0.2780,  0.1585],
        [ 0.1996,  0.2313,  0.1593,  0.1441,  0.5284,  0.6089,  0.4262,  0.4990,
         -0.0571, -0.5224, -0.7675, -0.3670],
        [ 0.1361,  0.1884,  0.1332,  0.4246,  0.5976,  0.4382,  0.4804,  0.5376,
          0.8509, -0.1591,  0.3079, -0.3948],
        [ 0.2459,  0.2737,  0.2902,  0.9892,  0.8445,  0.6455,  0.6013,  0.3996,
         -0.1203,  0.1149, -0.8852, -0.4345],
        [ 0.2554,  0.1024,  0.1243,  0.4442,  0.4847,  0.3753,  0.3542,  0.5288,
          0.1294,  0.7612, -0.4037, -0.4907],
        [ 0.1852,  0.2629,  0.1143,  0.8005,  0.9665,  0.4829,  0.4545,  0.5807,
          0.3983, -0.2366, -0.8171, -0.3431],
        [ 0.2660,  0.1470,  0.2511,  0.2668,  0.2718,  0.3558,  0.3467,  0.4545,
          0.1357,  0.3233, -0.8649, -0.3593],
        [ 0.1758,  0.2447,  0.1448,  0.1068,  0.3768,  0.6330,  0.4025,  0.5981,
          0.4034, -0.9041, -0.0815, -0.1145],
        [ 0.1334,  0.1594,  0.1117,  0.7234,  0.7051,  0.4303,  0.3712,  0.5394,
         -0.9855, -0.1327, -0.0569,  0.0895],
        [ 0.1205,  0.2326,  0.2777,  0.5483,  0.9702,  0.3524,  0.5263,  0.4284,
          0.3523, -0.1127, -0.3552, -0.8585],
        [ 0.2890,  0.2399,  0.2582,  0.8803,  0.4080,  0.4028,  0.4000,  0.3670,
         -0.0963,  0.3156,  0.9424,  0.0549],
        [ 0.1244,  0.1707,  0.2050,  0.5093,  0.2526,  0.5665,  0.4282,  0.5605,
          0.6121, -0.0320,  0.3257, -0.7199],
        [ 0.2130,  0.1083,  0.1996,  0.2209,  0.5531,  0.5501,  0.5381,  0.5670,
         -0.3280,  0.4695, -0.6633,  0.4818],
        [ 0.1998,  0.1674,  0.2467,  0.8034,  0.2889,  0.5178,  0.6429,  0.6091,
          0.0290, -0.7991,  0.5314, -0.2796],
        [ 0.1538,  0.2216,  0.1758,  0.9157,  0.8596,  0.3967,  0.4878,  0.6195,
          0.5293,  0.0706, -0.7950, -0.2878],
        [ 0.1271,  0.1388,  0.1781,  0.6055,  0.6471,  0.4492,  0.5019,  0.5206,
          0.5552, -0.4088, -0.3807, -0.6162],
        [ 0.2878,  0.2136,  0.2034,  0.7296,  0.8222,  0.5515,  0.3460,  0.3959,
          0.2475,  0.3209,  0.8976,  0.1736],
        [ 0.1693,  0.1401,  0.2480,  0.5930,  0.9853,  0.3543,  0.4828,  0.5380,
         -0.8285, -0.0754, -0.0749, -0.5498],
        [ 0.1535,  0.2843,  0.2577,  0.9704,  0.9133,  0.5776,  0.4093,  0.3733,
         -0.1003, -0.4884, -0.6963, -0.5164],
        [ 0.1168,  0.1042,  0.2698,  0.1223,  0.2900,  0.4281,  0.6561,  0.4106,
          0.2557,  0.4165, -0.8332, -0.2587],
        [ 0.2087,  0.2722,  0.2937,  0.1960,  0.9346,  0.6499,  0.4697,  0.6547,
         -0.4381, -0.1764,  0.8376, -0.2747],
        [ 0.2370,  0.2842,  0.2350,  0.5211,  0.7314,  0.3657,  0.6316,  0.3885,
         -0.2366, -0.4358, -0.3830,  0.7793],
        [ 0.1221,  0.2473,  0.2742,  0.8191,  0.4745,  0.5495,  0.6175,  0.5682,
         -0.5078, -0.0618, -0.8589, -0.0258],
        [ 0.2055,  0.1897,  0.2784,  0.5529,  0.8743,  0.5187,  0.6402,  0.5866,
          0.5061, -0.7978,  0.0086,  0.3275],
        [ 0.2589,  0.2107,  0.2886,  0.2819,  0.8018,  0.5727,  0.4977,  0.5167,
          0.1688, -0.4158,  0.0362,  0.8929],
        [ 0.2799,  0.1774,  0.1346,  0.8796,  0.9384,  0.6305,  0.5679,  0.3860,
         -0.2286,  0.8879, -0.3343, -0.2184],
        [ 0.2582,  0.1026,  0.2181,  0.4935,  0.6959,  0.6124,  0.3472,  0.4668,
          0.4542,  0.1331,  0.6144, -0.6312],
        [ 0.2634,  0.1360,  0.2881,  0.7654,  0.2847,  0.6367,  0.5711,  0.4283,
         -0.6848, -0.5008, -0.5269, -0.0502],
        [ 0.1064,  0.2766,  0.1063,  0.1339,  0.9209,  0.5230,  0.5823,  0.6049,
         -0.6852, -0.1264,  0.3632,  0.6186],
        [ 0.2823,  0.1928,  0.1359,  0.3902,  0.7718,  0.5545,  0.3924,  0.3647,
          0.2095, -0.1600, -0.8313,  0.4893],
        [ 0.1356,  0.2214,  0.1431,  0.7403,  0.8372,  0.4034,  0.4375,  0.5896,
         -0.7584, -0.0414, -0.6034, -0.2430],
        [ 0.2570,  0.2692,  0.2181,  0.5957,  0.5709,  0.5906,  0.4201,  0.4082,
          0.7313,  0.5486, -0.3786, -0.1445]], device='cuda:0')
tensor([[ 0.1630,  0.1170,  0.1343,  0.6863,  0.6626,  0.5960,  0.4090,  0.6172,
         -0.4237,  0.6347, -0.3787, -0.5236],
        [ 0.1917,  0.2286,  0.1711,  0.2439,  0.5095,  0.6019,  0.4181,  0.5152,
         -0.7787,  0.3488,  0.0312, -0.5205],
        [ 0.1293,  0.1235,  0.1828,  0.4384,  0.3096,  0.4346,  0.4808,  0.5618,
         -0.8571,  0.3437, -0.1400, -0.3572],
        [ 0.2592,  0.2608,  0.2323,  0.8135,  0.8656,  0.6431,  0.6031,  0.4286,
         -0.8127,  0.4474, -0.3436, -0.1458],
        [ 0.1531,  0.2296,  0.1248,  0.4013,  0.1570,  0.3599,  0.4024,  0.5185,
         -0.4208,  0.5940,  0.0135, -0.6855],
        [ 0.1987,  0.2375,  0.1179,  0.5858,  0.6243,  0.4729,  0.4545,  0.5758,
         -0.7421,  0.4574, -0.4521, -0.1888],
        [ 0.2624,  0.2540,  0.1852,  0.2622,  0.2963,  0.3386,  0.3398,  0.4476,
         -0.8438,  0.3232,  0.1379, -0.4056],
        [ 0.2566,  0.2324,  0.1855,  0.2029,  0.1093,  0.6408,  0.4319,  0.4983,
         -0.3603,  0.7097, -0.2401, -0.5557],
        [ 0.1639,  0.1353,  0.1494,  0.4947,  0.5564,  0.4326,  0.3827,  0.4977,
         -0.5882,  0.4781, -0.5506, -0.3498],
        [ 0.1519,  0.2057,  0.2480,  0.3223,  0.5185,  0.3922,  0.5507,  0.4446,
         -0.4215,  0.8022, -0.4022, -0.1307],
        [ 0.2337,  0.2768,  0.2423,  0.7401,  0.4105,  0.4054,  0.4052,  0.3858,
         -0.6201,  0.6997, -0.3093, -0.1740],
        [ 0.1210,  0.1677,  0.1872,  0.3454,  0.2310,  0.5654,  0.4316,  0.5811,
         -0.7153,  0.3179,  0.0071, -0.6223],
        [ 0.2184,  0.1870,  0.2020,  0.2726,  0.2964,  0.5651,  0.5283,  0.5028,
         -0.6453,  0.2344, -0.0224, -0.7267],
        [ 0.1901,  0.2181,  0.2076,  0.4778,  0.4167,  0.5146,  0.6539,  0.6032,
         -0.3854,  0.7534, -0.1189, -0.5194],
        [ 0.1909,  0.1615,  0.2119,  0.6875,  0.6855,  0.3861,  0.5104,  0.5994,
         -0.5495,  0.6275, -0.2012, -0.5137],
        [ 0.1729,  0.1322,  0.1732,  0.4193,  0.5756,  0.4634,  0.5046,  0.4908,
         -0.4382,  0.5721, -0.5471, -0.4259],
        [ 0.2239,  0.2710,  0.2052,  0.7652,  0.6803,  0.5463,  0.3632,  0.4036,
         -0.5316,  0.7329, -0.0942, -0.4139],
        [ 0.2065,  0.1659,  0.2329,  0.3605,  0.6712,  0.3505,  0.5146,  0.4863,
         -0.6230,  0.5484, -0.4425, -0.3395],
        [ 0.1456,  0.2459,  0.2273,  0.7667,  0.7134,  0.5705,  0.4229,  0.4159,
         -0.8172,  0.4838, -0.0928, -0.2992],
        [ 0.1174,  0.2648,  0.1561,  0.2703,  0.1541,  0.4152,  0.6272,  0.3920,
         -0.8950,  0.3129, -0.0259, -0.3168],
        [ 0.2603,  0.2921,  0.2661,  0.4779,  0.2944,  0.6103,  0.4599,  0.6307,
         -0.8729,  0.3918,  0.0788, -0.2799],
        [ 0.2762,  0.2781,  0.2288,  0.4289,  0.6837,  0.3454,  0.6506,  0.3698,
         -0.7695,  0.3812, -0.4475, -0.2495],
        [ 0.2332,  0.2448,  0.2165,  0.4134,  0.4686,  0.5900,  0.6019,  0.5035,
         -0.7365,  0.6298, -0.2319,  0.0842],
        [ 0.1957,  0.1801,  0.2783,  0.5315,  0.6489,  0.5190,  0.6397,  0.6006,
         -0.5024,  0.7965, -0.0535, -0.3320],
        [ 0.2499,  0.2885,  0.2022,  0.5052,  0.3483,  0.5616,  0.5026,  0.5365,
         -0.7641,  0.2749, -0.2994, -0.5010],
        [ 0.1688,  0.2537,  0.1379,  0.7182,  0.6605,  0.6098,  0.5630,  0.3986,
         -0.7925,  0.4795, -0.1291, -0.3540],
        [ 0.2642,  0.1096,  0.2140,  0.5459,  0.6507,  0.6209,  0.3557,  0.4677,
         -0.6275,  0.6159, -0.1721, -0.4442],
        [ 0.1646,  0.2522,  0.2490,  0.2561,  0.5137,  0.6093,  0.5871,  0.4333,
         -0.8367,  0.3655, -0.2208, -0.3430],
        [ 0.2428,  0.1152,  0.1397,  0.4883,  0.4749,  0.5474,  0.5982,  0.6099,
         -0.4195,  0.4782, -0.4393, -0.6343],
        [ 0.1848,  0.1716,  0.2682,  0.4392,  0.3136,  0.5784,  0.4008,  0.3472,
         -0.6743,  0.3131, -0.0589, -0.6662],
        [ 0.2163,  0.1408,  0.2062,  0.6765,  0.5591,  0.3840,  0.4244,  0.5450,
         -0.4820,  0.5742, -0.3126, -0.5833],
        [ 0.2500,  0.2579,  0.2323,  0.6857,  0.4722,  0.5751,  0.4233,  0.4227,
         -0.5385,  0.7467,  0.1290, -0.3685]], device='cuda:0',
       grad_fn=<CatBackward>)

"""