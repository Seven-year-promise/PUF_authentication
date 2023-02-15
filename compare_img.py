# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage import io
import cv2
import numpy as np
from config import *
import render_bg
from PixelWiseSim import Pixel_Wise_Sim
import matplotlib.pyplot as plt

# rember, important, the saving by io.imsave can change the distribution of the iamges

def compare():
    im_for_train = io.imread(RENDER_BG_SAVE_ALL / "7.jpg")
    im_for_test = io.imread(TEST_SAVEL_ALL_SMALLER / "brightness/7_15.jpg", as_gray=True)
    im_for_test = np.array(im_for_test * 255, np.uint8)
    im_for_test = render_bg.processing(im_for_test)

    io.imsave(TEST_SAVEL_ALL_SMALLER / "rendered/7_15.jpg", im_for_test)

    im_for_test = io.imread(TEST_SAVEL_ALL_SMALLER / "rendered/7_15.jpg")


    sim = Pixel_Wise_Sim(im_for_test, im_for_train, 1)
    diff = np.array(np.abs(im_for_test-im_for_train), np.uint8)
    im_for_train = np.array(im_for_train, np.uint8)
    im_for_test = np.array(im_for_test, np.uint8)

    histr1 = cv2.calcHist([im_for_train], [0], None, [256], [0, 256])
    histr2 = cv2.calcHist([im_for_test], [0], None, [256], [0, 256])

    plt.plot(histr1)
    plt.plot(histr2)
    plt.show()
    print(sim)
    cv2.imshow("train", im_for_train)
    cv2.imshow("test", im_for_test)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)


    print(im_for_test.max(), im_for_train.max())


if __name__ == '__main__':
    compare()