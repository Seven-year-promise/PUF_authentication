# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage import io, draw, color, transform
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.exposure import adjust_gamma
from skimage import transform
from skimage.util import random_noise
import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from util import circle_center, circle_center_otsu, circle_area_otsu, padding, find_edge_lw, find_edge_rw
import cv2
from config import *

half_size = 960
# Press the green button in the gutter to run the script.
rotation_angles = np.arange(180) * (360/180)
rotation_angles = rotation_angles.tolist()

rotation_angles = rotation_angles[1:15] + \
                  rotation_angles[16:30] + \
                  rotation_angles[31:45] + \
                  rotation_angles[46:60] + \
                  rotation_angles[61:75] + \
                  rotation_angles[76:90] + \
                  rotation_angles[91:105] + \
                  rotation_angles[106:120] + \
                  rotation_angles[121:135] + \
                  rotation_angles[136:150] + \
                  rotation_angles[151:165] + \
                  rotation_angles[166:]

gaussian_sigma = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

noise_var = 0.002

brightness = []

resolutions = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

test_save_path = TEST_SAVEL_ALL_SMALLER
def generate():
    im_cnt = 0
    all_im_paths = RGB_SAVE_ALL.rglob("*.jpg")

    def get_order(file):
        return int(file.name.split(".")[0])

    all_im_paths = list(sorted(all_im_paths, key=get_order))
    print(all_im_paths)

    annotations = {}
    for im_p in all_im_paths:

        im_rgb = io.imread(im_p, as_gray=False)
        im_name = im_p.name.split(".")[0]
        if int(im_name) > 1299:
            label = -1
        else:
            label = int(im_name)


        this_im_cnt = 0

        # for rotation
        slected_angles = random.sample(rotation_angles, 10)
        for angle in slected_angles:
            im_rgb_copy = im_rgb.copy()
            im_rgb_rotated = transform.rotate(im_rgb_copy, angle, mode="edge")
            #im_rgb_rotated = random_noise(im_rgb_rotated, mode='gaussian', var=noise_var, seed=None, clip=True)

            im_rotated_path = str(test_save_path / "rotation" / (im_name + "_" + str(this_im_cnt) + ".jpg"))
            annotations[im_cnt] = {}
            annotations[im_cnt]["PUF"] = im_rotated_path
            annotations[im_cnt]["height"] = 768
            annotations[im_cnt]["width"] = 768
            annotations[im_cnt]["rotation"] = angle
            annotations[im_cnt]["blur"] = "None"
            annotations[im_cnt]["brightness"] = "None"
            annotations[im_cnt]["noise"] = "None"
            annotations[im_cnt]["resolution"] = "None"
            annotations[im_cnt]["label"] = label

            io.imsave(im_rotated_path, im_rgb_rotated)

            this_im_cnt += 1
            im_cnt += 1

        # for brightness
        for i in range(10):
            gamma = np.random.uniform(0.5, 1.5)
            im_rgb_copy = im_rgb.copy()
            im_rgb_brightness = adjust_gamma(im_rgb_copy, gamma)

            im_brightness_path = str(test_save_path / "brightness" / (im_name + "_" + str(this_im_cnt) + ".jpg"))
            annotations[im_cnt] = {}
            annotations[im_cnt]["PUF"] = im_brightness_path
            annotations[im_cnt]["height"] = 768
            annotations[im_cnt]["width"] = 768
            annotations[im_cnt]["rotation"] = "None"
            annotations[im_cnt]["blur"] = "None"
            annotations[im_cnt]["brightness"] = "gamma_" + str(gamma)
            annotations[im_cnt]["noise"] = "None"
            annotations[im_cnt]["resolution"] = "None"
            annotations[im_cnt]["label"] = label

            io.imsave(im_brightness_path, im_rgb_brightness)

            this_im_cnt += 1
            im_cnt += 1

        # for blurring
        """
        for g_s in gaussian_sigma:
            im_rgb_copy = im_rgb.copy()
            im_rgb_blur = gaussian(im_rgb_copy, sigma=g_s)
            im_blur_path = str(test_save_path / "blur" / (im_name + "_" + str(this_im_cnt) + ".jpg"))
            annotations[im_cnt] = {}
            annotations[im_cnt]["PUF"] = im_blur_path
            annotations[im_cnt]["height"] = 768
            annotations[im_cnt]["width"] = 768
            annotations[im_cnt]["rotation"] = "None"
            annotations[im_cnt]["blur"] = "gaussian_sigma_" + str(g_s)
            annotations[im_cnt]["brightness"] = "None"
            annotations[im_cnt]["noise"] = "None"
            annotations[im_cnt]["resolution"] = "None"
            annotations[im_cnt]["label"] = label

            io.imsave(im_blur_path, im_rgb_blur)

            this_im_cnt += 1
            im_cnt += 1

        

        # for noise
        for i in range(10):
            im_rgb_copy = im_rgb.copy()
            im_rgb_noise = random_noise(im_rgb_copy, mode='gaussian', var=noise_var, seed=None, clip=True)

            im_noise_path = str(test_save_path / "noise" / (im_name + "_" + str(this_im_cnt) + ".jpg"))
            annotations[im_cnt] = {}
            annotations[im_cnt]["PUF"] = im_noise_path
            annotations[im_cnt]["height"] = 768
            annotations[im_cnt]["width"] = 768
            annotations[im_cnt]["rotation"] = "None"
            annotations[im_cnt]["blur"] = "None"
            annotations[im_cnt]["brightness"] = "None"
            annotations[im_cnt]["noise"] = "gaussian_var_"+str(noise_var)
            annotations[im_cnt]["resolution"] = "None"
            annotations[im_cnt]["label"] = label

            io.imsave(im_noise_path, im_rgb_noise)

            this_im_cnt += 1
            im_cnt += 1

            # for resolutions
        for re in resolutions:
            im_rgb_copy = im_rgb.copy()
            im_rgb_resolution = transform.resize(im_rgb_copy, (re, re))
            im_resolution_path = str(test_save_path / "resolution" / (im_name + "_" + str(this_im_cnt) + ".jpg"))

            annotations[im_cnt] = {}
            annotations[im_cnt]["PUF"] = im_resolution_path
            annotations[im_cnt]["height"] = re
            annotations[im_cnt]["width"] = re
            annotations[im_cnt]["rotation"] = "None"
            annotations[im_cnt]["blur"] = "None"
            annotations[im_cnt]["brightness"] = "None"
            annotations[im_cnt]["noise"] = "None"
            annotations[im_cnt]["resolution"] = str(re)
            annotations[im_cnt]["label"] = label

            io.imsave(im_resolution_path, im_rgb_resolution)

            this_im_cnt += 1
            im_cnt += 1
        """
    with open(TEST_PATH / "test_smaller.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)



    """
            for m_s in median_size:
                im_rgb_copy = im_rgb.copy()
                im_rgb_blur = np.array([median(im_rgb_copy[0], disk(m_s)), median(im_rgb_copy[1], disk(m_s)), median(im_rgb_copy[2], disk(m_s))])
                im_blur_path = str(test_save_path / "blur" / (im_name + "_" + str(this_im_cnt) + ".jpg"))
                annotations[im_cnt] = {}
                annotations[im_cnt]["PUF"] = im_blur_path
                annotations[im_cnt]["height"] = 768
                annotations[im_cnt]["width"] = 768
                annotations[im_cnt]["rotation"] = "None"
                annotations[im_cnt]["blur"] = "median_"+str(m_s)
                annotations[im_cnt]["brightness"] = "None"
                annotations[im_cnt]["noise"] = "None"
                annotations[im_cnt]["resolution"] = "None"
                annotations[im_cnt]["label"] = int(im_name)

                io.imsave(im_blur_path, im_rgb_blur)

                this_im_cnt += 1
                im_cnt += 1
    """
if __name__ == '__main__':
    generate()