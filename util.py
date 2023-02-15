from skimage.filters import threshold_otsu
from skimage import io, draw, color, transform
import numpy as np

def binarize(im, th):
    binary = im > th

    return binary

def circle_center(im, thre = 0.25):
    gray = color.rgb2gray(im) # change pixels tp 0 - 1
    coors = np.where(gray>thre)
    y_ave = int(np.average(coors[0]))
    x_ave = int(np.average(coors[1]))

    gray[np.where(gray<thre)] = 0

    return [y_ave, x_ave], gray

def circle_center_otsu(im):
    gray = color.rgb2gray(im) # change pixels tp 0 - 1
    thresh = threshold_otsu(gray)
    coors = np.where(gray>thresh)
    y_ave = int(np.average(coors[0]))
    x_ave = int(np.average(coors[1]))

    gray[np.where(gray<thresh)] = 0

    otsu_im = (gray>thresh) * 1.0

    return [y_ave, x_ave], gray, otsu_im

def circle_area_otsu(im):
    gray = color.rgb2gray(im) # change pixels tp 0 - 1
    thresh = threshold_otsu(gray)
    coors = np.where(gray>thresh)
    y_min = np.min(coors[0])-1
    x_min = np.min(coors[1])-1

    y_max = np.max(coors[0]) + 2
    x_max = np.max(coors[1]) + 2

    gray[np.where(gray<thresh)] = 0

    otsu_im = (gray>thresh) * 1.0

    return [y_min, x_min, y_max, x_max], gray, otsu_im

def padding(im, half_size):
    old_h, old_w = im.shape
    left = right= int(half_size-old_w/2)
    top = bottom = int(half_size-old_h/2)
    assert left >= 0
    assert right >= 0
    assert top >= 0
    assert bottom >= 0

    new_im = np.zeros((old_h+top+bottom, old_w+left+right), dtype = float)
    new_im[top:top+old_h, left:left+old_w] = im

    return new_im

def center_by_boundary(binary):
    coordinates = np.where(binary>0)
    #print(coordinates[0], coordinates[1])
    min_x = np.min(coordinates[0])
    max_x = np.max(coordinates[0])
    min_y = np.min(coordinates[1])
    max_y = np.max(coordinates[1])

    return int((min_y + max_y) / 2.0), int((min_x + max_x) / 2.0)

def find_edge_lw(im_without_bg):
    #print(im_without_bg.shape[1])
    for w in range(im_without_bg.shape[1]):
        if np.sum(im_without_bg[:, w:w+20]) < 1/255:
            if np.sum(im_without_bg[:, w+50:w+70]) < 1/255:
                return w

    return 0

def find_edge_rw(im_without_bg):
    width = im_without_bg.shape[1]
    for w in range(width):
        #print(w)
        if np.sum(im_without_bg[:, -(w+21):-(w+1)]) < 1/255:
            if np.sum(im_without_bg[:, -(w+70):-(w+50)]) < 1/255:
                return width - w - 10

    return width