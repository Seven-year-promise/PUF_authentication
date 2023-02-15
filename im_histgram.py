from __future__ import print_function, division
import os
import argparse
import time
import cv2
import pickle
import torchvision as tv, torchvision.transforms as tr
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from ebm import EBM
import torchvision.transforms as transforms
# from autoencoder import Autoencoder
from autoencoder_alternate import Autoencoder
# import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc
from Dataloader import MyRotationTransform, JEMDataset
from config import *
from skimage import io, draw, color, transform, exposure
import csv

im_id = 1299

if __name__ == "__main__":
    im_ori_path = ORI_SAVE_ALL / (str(im_id) + ".jpg")
    im_render_bg_path = RENDER_BG_SAVE_ALL / (str(im_id) + ".jpg")
    im_ori = io.imread(im_ori_path)
    im_render_bg = io.imread(im_render_bg_path)

    histr = cv2.calcHist([im_ori], [0], None, [256], [0, 256])
    #print(histr[:, 0])
    #plt.bar(np.arange(256), histr[:, 0])
    #plt.show()
    #plt.clf()

    with open(PAPER_SAVE_PATH/("im" + str(im_id) + "_ori_histgram.csv"), 'w', newline='') as f:
        histriter = csv.writer(f, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        histriter.writerows(histr)



    histr2 = cv2.calcHist([im_render_bg], [0], None, [256], [0, 256])
    #print(histr2[:, 0])
    #plt.bar(np.arange(256), histr2[:, 0])
    #plt.show()

    with open(PAPER_SAVE_PATH / ("im" + str(im_id) + "_render_bg_histgram.csv"), 'w', newline='') as f:
        histriter = csv.writer(f, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        histriter.writerows(histr2)