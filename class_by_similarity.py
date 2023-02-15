# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage import io, draw, color, transform
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
from skimage.metrics import structural_similarity

def binarize(im, th):
    binary = im > th

    return binary

def circle_center(im, thre):
    gray = color.rgb2gray(im) # change pixels tp 0 - 1

    gray_ave = np.average(gray)
    coors = np.where(gray>thre)
    th = (gray > thre) * 1
    print(gray)
    print(th)
    x_ave = int(np.average(coors[0]))
    y_ave = int(np.average(coors[1]))
    rr, cc = draw.ellipse_perimeter(x_ave, y_ave, 30, 30)
    im[rr, cc] = [255, 0, 0]

    fig, (ax0, ax1) = plt.subplots(1, 2)
    #ax0.imshow(gray, cmap='gray')
    ax0.imshow(im)
    ax1.imshow(th, cmap='gray')

    plt.show()

def get_similarity(im1, im2):
    im_ft_1 = fft2(im1)
    im_ft_2 = fft2(im2)

    #fig, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.imshow(gray, cmap='gray')
    #ax0.imshow(im_ft_1, cmap='gray')
    #ax1.imshow(im_ft_2, cmap='gray')

    #plt.show()

    return structural_similarity(im1, im2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "data/test/dataset/im/"
    file_names = os.listdir(file_path)
    N = len(file_names)
    fixed_im = io.imread(file_path + "ture_0.0.jpg")
    correct_pred = 0
    total_pred = 0
    fixed_im = transform.resize(fixed_im, (1400, 1400))
    for j, file_name_j in enumerate(file_names):
        im_j = io.imread(file_path + file_name_j)  # output height x width x channel, 0-255, RGB
        im_j = transform.resize(im_j, (1400, 1400))
        print(file_name_j, fixed_im.shape, im_j.shape)
        sm = get_similarity(im1=fixed_im, im2=im_j)
        if file_name_j[0] == "t":
            label = 1
        else:
            label = 0
        prediction = (sm>0.8) *1
        print(sm, label, prediction)
        if label == prediction:
            print("yes")
            correct_pred += 1
        total_pred += 1

    accuracy = (correct_pred *100.0) / total_pred
    print("accuracy is: ", accuracy)