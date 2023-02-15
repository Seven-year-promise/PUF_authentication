# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage import io, draw, color, transform
import os
import numpy as np
import matplotlib.pyplot as plt
from util import circle_center, circle_center_otsu, center_by_boundary
import cv2

half_size = 900
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "data/ori/sixth/training images/for_test/"
    file_names = os.listdir(file_path)
    save_path = "data/add/test/external_real_test/"
    for _, f_n in enumerate(file_names):
        print(file_path + f_n)
        types = os.listdir(file_path + f_n)
        j = 0
        for t in types:
            im_names = os.listdir(os.path.join(file_path, f_n, t))
            for i_n in im_names:
                i_path = os.path.join(file_path, f_n, t, i_n) #"data/ori/sixth/training images/for_test/10/rotation/0136.136.jpg" #
                print("reading: " + i_path)
                im = io.imread(i_path) # output height x width x channel, 0-255, RGB
                im = transform.resize(im, (1900, 1900))
                (y_c, x_c), im_gray, otsu_im = circle_center_otsu(im)
                if y_c-half_size < 0 or x_c-half_size <0 or y_c+half_size >= 1900 or x_c+half_size >= 1900:
                    y_c2, x_c2 = center_by_boundary(otsu_im)
                    y_c, x_c = int((y_c + y_c2)/2.0), int((x_c + x_c2)/2.0)
                #cv2.imshow("otsu", otsu_im)
                #cv2.waitKey(0)
                print(y_c, x_c)
                cropped = im_gray[(y_c-half_size):(y_c+half_size), (x_c-half_size):(x_c+half_size)]
                cropped = transform.resize(cropped, (half_size*2, half_size*2))
                cropped_otsu_im = otsu_im[(y_c-half_size):(y_c+half_size), (x_c-half_size):(x_c+half_size)]
                cropped_otsu_im = transform.resize(cropped_otsu_im, (half_size * 2, half_size * 2))
                cropped_uint = np.asarray(cropped * 255, np.uint8)
                cropped_otsu_im_uint = np.asarray(cropped_otsu_im * 255, np.uint8)

                im_save_name = f_n + "_" + str(j) + ".jpg"
                io.imsave(save_path + "im/" + im_save_name, cropped_uint)
                print(im_save_name)
                im_save_name = f_n + "_" + str(j) + "_otsu.jpg"
                io.imsave(save_path + "binary/" + im_save_name, cropped_otsu_im_uint)
                print(im_save_name)
                j += 1