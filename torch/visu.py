import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from time import time, sleep
from quaternion import rotate, mat_from_quaternion_np, conjugate_np, multiply_np, to_magnitude_np, conjugate, multiply, to_magnitude
from helpers import plot_render, plot_grad_flow, plot_points, norm_img, randquat, slerp
from classes import ChamferQuatLoss, IoUAccuracy, ChamferLoss
import torch
from sklearn.preprocessing import normalize

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

    granularity = 32
    #loss = ChamferQuatLoss(render_size=granularity, device=device)
    loss = ChamferLoss(render_size=granularity, device=device)
    acc = IoUAccuracy(render_size=128, device=device)

    torch.autograd.set_detect_anomaly(True)

    q_true = randquat()
    q_pred = np.array([0, 0, 0, 1.])

    # Create SQ pointcloud
    #params = [0.23, 0.11, 0.29, 0.1, 0.1]
    params_true = randsq()
    print(params_true)
    params_pred = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]

    pts = sample_points_uniform(params_true, rad_resolution=0.1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Create convex hull
    hull, _ = pcd.compute_convex_hull()
    hull.compute_vertex_normals()

    # Line mesh for true quadric
    hull_ls1 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls2 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls3 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls4 = o3d.geometry.LineSet.create_from_triangle_mesh(hull)

    hull_ls1.paint_uniform_color((1, 0, 0))
    hull_ls2.paint_uniform_color((0, 1, 0))
    hull_ls3.paint_uniform_color((0, 0, 1))
    hull_ls4.paint_uniform_color((0, 1, 1))

    print(q_true)
    print(mat_from_quaternion_np(conjugate_np(multiply_np(q_true, [0, 0, 0, 1])))[0])
    print(multiply_np(q_true, [1, 0, 0, 0]))
    print(mat_from_quaternion_np(conjugate_np(multiply_np(q_true, [1, 0, 0, 0])))[0])
    print(multiply_np(q_true, [0, 1, 0, 0]))
    print(mat_from_quaternion_np(conjugate_np(multiply_np(q_true, [0, 1, 0, 0])))[0])
    print(multiply_np(q_true, [0, 0, 1, 0]))
    print(mat_from_quaternion_np(conjugate_np(multiply_np(q_true, [0, 0, 1, 0])))[0])

    # TODO: is rotation without conjugation correct?
    hull_ls1.rotate(mat_from_quaternion_np(q_true)[0], [0, 0, 0])
    hull_ls2.rotate(mat_from_quaternion_np(multiply_np(q_true, [0, 1, 0, 0]))[0], [0, 0, 0])
    hull_ls3.rotate(mat_from_quaternion_np(multiply_np(q_true, [1, 0, 0, 0]))[0], [0, 0, 0])
    hull_ls4.rotate(mat_from_quaternion_np(multiply_np(q_true, [0, 0, 1, 0]))[0], [0, 0, 0])
    #exit()
    vis = init_visualization()

    vis.add_geometry(hull_ls1)
    #vis.add_geometry(hull_ls2)
    #vis.add_geometry(hull_ls3)
    #vis.add_geometry(hull_ls4)


    while True:
        pts = sample_points_uniform(params_pred, rad_resolution=0.1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Create convex hull
        hull_pred, _ = pcd.compute_convex_hull()
        hull_pred.compute_vertex_normals()
        vis.add_geometry(hull_pred)


        mat_q = mat_from_quaternion_np(q_pred)[0]
        hull_pred.rotate(mat_q, [0, 0, 0])

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
        a = acc(true, pred[:, 8:])

        diff = to_magnitude(multiply(torch.from_numpy(q_true), conjugate(torch.from_numpy(q_pred))))
        loss_np = l.detach().cpu().item()
        acc_np = a.detach().cpu().item()

        l.backward()

        print("Predicted:", q_pred)
        print("Predicted:", normalize([q_pred]))
        print("Grads:", pred.grad)
        print("Loss:", loss_np)
        print("Accuracy:", acc_np)
        print("Angle difference:", diff)
        print("Possible True:")
        print("---------------------------------------")
        lr = 0.002

        gradient = pred.grad.detach().cpu().numpy()[0]
        #print(gradient)
        #exit()
        q_pred -= lr * gradient[8:]
        q_pred = normalize([q_pred])[0]

        params_pred -= lr * gradient[:8]


        sleep(0.1)
        vis.update_geometry(hull_pred)
        vis.poll_events()
        vis.update_renderer()
        #hull_pred.rotate(mat_q.T, [0, 0, 0])
        vis.remove_geometry(hull_pred)

    vis.destroy_window()
