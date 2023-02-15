from skimage import io, draw, color, transform
from skimage.filters import median, gaussian
import numpy as np
import math
from skimage.morphology import disk
import cv2
from SimISSM import fsim_torch
from piq import fsim as fsim_fast
import torch
import time

def sim(im, target):
    coors = np.where(target > 0)
    matached = np.where(im[coors] > 0)
    return matached[0].shape[0] / coors[0].shape[0]

def sim_norm(im, target):
    coors = np.where(target > 0)
    diff = np.abs(im - target)[0, :, :]
    diff = diff[coors].mean()
    return 1 - diff / 255.0

def Pixel_Wise_Sim(im, target, angle_thre):
    #im = median(im, disk(3))
    #target = median(target, disk(3))
    im_show = np.array(im * 255.0, np.uint8)
    cv2.imshow("show_ori", im_show)
    sim_0 = sim(im, target)
    im_rotate_m90 = transform.rotate(im, -90)
    sim_m90 = sim(im_rotate_m90, target)
    im_rotate_90 = transform.rotate(im, 90)
    sim_90 = sim(im_rotate_90, target)
    im_rotate_180 = transform.rotate(im, 180)
    sim_180 = sim(im_rotate_180, target)

    a1 = [0, 180]
    a2 = [-90, 90]

    sim_pairs_1 = [sim_0, sim_180]
    sim_pairs_2 = [sim_m90, sim_90]

    angle_area = [a1[(sim_0 < sim_180) * 1], a2[(sim_m90 < sim_90) * 1]]
    if angle_area[0] == -90 and angle_area[1] == 180:
        angle_area[0] = 270

    angle_step = 90
    old_edge_sim1 = sim_pairs_1[(sim_0 < sim_180) * 1]
    old_edge_sim2 = sim_pairs_2[(sim_m90 < sim_90) * 1]
    new_angle, sim_new = 0, 0
    while angle_step > angle_thre:

        new_angle = (angle_area[0] + angle_area[1]) / 2.0
        im_rotate_new = transform.rotate(im, new_angle)

        im_rotate_new_show = np.array(im_rotate_new * 255.0, np.uint8)
        cv2.imshow("show_rotaed", im_rotate_new_show)
        cv2.waitKey(0)

        sim_new = sim(im_rotate_new, target)

        if old_edge_sim1 > old_edge_sim2:
            old_edge_sim2 = sim_new
            angle_area[1] = new_angle
        else:
            old_edge_sim1 = sim_new
            angle_area[0] = new_angle


        angle_step = np.abs(angle_area[0] - angle_area[1])
        #print(angle_area, new_angle, angle_step, sim_new, old_edge_sim1, old_edge_sim2)


    return new_angle, np.max([old_edge_sim1, old_edge_sim2])


def iterated_fsim(im, target, angle_thre):
    im = color.gray2rgb(im*255)#np.expand_dims(im, 2)
    target = color.gray2rgb(target)#np.expand_dims(target, 2)
    #im = median(im, disk(3))
    #target = median(target, disk(3))
    #print(np.max(im), np.min(im), np.max(target), np.min(target))
    #im_show = np.array(im * 255.0, np.uint8)
    #cv2.imshow("show_ori", im_show)
    #print(target.shape, im.shape)
    sim_0 = fsim_torch(target, im, data_range=1.0)
    im_rotate_m90 = transform.rotate(im, -90)
    sim_m90 = fsim_torch(target, im_rotate_m90, data_range=1.0)
    im_rotate_90 = transform.rotate(im, 90)
    sim_90 = fsim_torch(target, im_rotate_90, data_range=1.0)
    im_rotate_180 = transform.rotate(im, 180)
    sim_180 = fsim_torch(target, im_rotate_180, data_range=1.0)

    a1 = [0, 180]
    a2 = [-90, 90]

    sim_pairs_1 = [sim_0, sim_180]
    sim_pairs_2 = [sim_m90, sim_90]

    angle_area = [a1[(sim_0 < sim_180) * 1], a2[(sim_m90 < sim_90) * 1]]
    if angle_area[0] == -90 and angle_area[1] == 180:
        angle_area[0] = 270

    angle_step = 90
    old_edge_sim1 = sim_pairs_1[(sim_0 < sim_180) * 1]
    old_edge_sim2 = sim_pairs_2[(sim_m90 < sim_90) * 1]
    new_angle, sim_new = 0, 0
    while angle_step > angle_thre:

        new_angle = (angle_area[0] + angle_area[1]) / 2.0
        im_rotate_new = transform.rotate(im, new_angle)
        #print(im_rotate_new.shape)

        #im_rotate_new_show = np.array(im_rotate_new * 255.0, np.uint8)
        #cv2.imshow("show_rotaed", im_rotate_new_show)
        #cv2.waitKey(0)

        sim_new = fsim_torch(target, im_rotate_new, data_range=1.0)

        if old_edge_sim1 > old_edge_sim2:
            old_edge_sim2 = sim_new
            angle_area[1] = new_angle
        else:
            old_edge_sim1 = sim_new
            angle_area[0] = new_angle


        angle_step = np.abs(angle_area[0] - angle_area[1])
        #print(angle_area, new_angle, angle_step, sim_new, old_edge_sim1, old_edge_sim2)


    return new_angle, np.max([old_edge_sim1, old_edge_sim2])

def fsim_360(im, target, angle_thre=5, batch_size = 72, device="cuda:0"):
    im = color.gray2rgb(im * 255)  # np.expand_dims(im, 2)
    target = color.gray2rgb(target)  # np.expand_dims(target, 2)
    all_sims = []
    all_rotated_imags = []
    #print(time.time())
    for i in range(int(360/angle_thre)):
        angle = i*angle_thre
        im_rotate_new = transform.rotate(im, angle)
        all_rotated_imags.append(im_rotate_new)
    #print(time.time())
    batches = len(all_rotated_imags) // batch_size
    target_tensor = torch.from_numpy(target).double().permute(2, 0, 1).unsqueeze(dim=0)
    target_tensor_r1 = target_tensor.repeat(batch_size, 1, 1, 1).to(device)

    for i in range(batches):
        im_batches = np.array(all_rotated_imags[(i*batch_size):((i+1)*batch_size)])
        im_batches_tensor = torch.from_numpy(im_batches).double().permute(0, 3, 1, 2).to(device)
        #print(im_batches_tensor.size(), target_tensor.size())
        index = fsim_fast(im_batches_tensor, target_tensor_r1, reduction="none", data_range=1.0)
        all_sims += index.detach().cpu().numpy().tolist()
    if len(all_rotated_imags) > batch_size*batches:
        target_tensor_r2 = target_tensor.repeat(len(all_rotated_imags) - batch_size*batches, 1, 1, 1).to(device)
        last_batch = np.array(all_rotated_imags[(batches*batch_size):])
        last_batch_tensor = torch.from_numpy(last_batch).double().permute(0, 3, 1, 2).to(device)
        last_index = fsim_fast(last_batch_tensor, target_tensor_r2, reduction="none", data_range=1.0)
        all_sims += last_index.detach().cpu().numpy().tolist()
        #print(time.time())

    return None, np.max(all_sims)

def Pixel_Wise_Sim_360(im, target, angle_thre=5):
    all_sims = []
    for i in range(int(360/angle_thre)):
        angle = i*angle_thre
        im_rotate_new = transform.rotate(im, angle)

        sim_new = sim(im_rotate_new, target)

        all_sims.append(sim_new)

    return None, np.max(all_sims)

if __name__ == "__main__":
    im_1 = io.imread("./data/add/previous/binary/4_otsu.jpg")
    im_2 = io.imread("./data/add/test/external_real_test/binary/015_otsu.jpg")
    #im_2 = io.imread("./data/add/test/fake_test/binary/0_otsu.jpg")
    angle_rotated, similarity = Pixel_Wise_Sim(im_1, im_2, 1)

    print(angle_rotated, similarity)