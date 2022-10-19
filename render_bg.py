# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage import io, draw, color, transform
import os
import numpy as np
import matplotlib.pyplot as plt
from util import circle_center, circle_center_otsu, circle_area_otsu, padding, find_edge_lw, find_edge_rw
import cv2
from config import *

half_size = 960
# Press the green button in the gutter to run the script.
def render_bg():
    for im_p in list(ORI_SAVE_ALL.rglob("*.jpg")):

        #print(im_p.name)
        #if id_cnt < 1481:
        #    continue
        im = io.imread(im_p)
        histr = cv2.calcHist([im], [0], None, [256], [0, 256])

        im = np.array(im, np.float)

        bg_value = np.argmax(histr)

        im = im-bg_value
        im[np.where(im < 0)] = 0

        im = im * 255.0 / (255-bg_value)

        max_v = np.max(im)

        im = im / (max_v/255.0)

        print(np.argmax(histr), (255/(255-bg_value)), max_v)

        #im = np.array(im, np.uint8)

        io.imsave(RENDER_BG_SAVE_ALL / im_p.name, im)


if __name__ == '__main__':
    render_bg()