import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import random
import json
import numpy as np
import os
import math
import csv
import sys
from skimage import io, draw, color, transform, exposure
from skimage.util import random_noise
import numpy as np
from PIL import Image, ImageOps
import render_bg
import matplotlib.pyplot as plt
import cv2
from config import *


def brightness(x, gamma=1, gain=1, is_random=True):
    """Change the brightness of a single image, randomly or non-randomly.

    Parameters
    -----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    gamma : float, small than 1 means brighter.
        Non negative real number. Default value is 1.

        - If is_random is True, gamma in a range of (1-gamma, 1+gamma).
    gain : float
        The constant multiplier. Default value is 1.
    is_random : boolean, default False
        - If True, randomly change brightness.

    References
    -----------
    - `skimage.exposure.adjust_gamma <http://scikit-image.org/docs/dev/api/skimage.exposure.html>`_
    - `chinese blog <http://www.cnblogs.com/denny402/p/5124402.html>`_
    """
    if is_random:
        gamma = np.random.uniform(1-gamma, 1+gamma)
    x = exposure.adjust_gamma(x, gamma, gain)
    return x

class CLassOnlyTestDataset(data.Dataset):

    def __init__(self, first_classes=1500, cropped_size=None, transform=[], anno_path="", im_type="PUF"):
        self.cropped_size = cropped_size
        self.transform = transform

        anno_file = open(anno_path)
        anno_data = json.load(anno_file)
        im_paths = []
        labels = []

        for k in anno_data.keys():
            label = anno_data[k]["label"]
            #if anno_data[k]["brightness"] != "None":
            #    continue
            if label >= first_classes:
                continue
            #if label < 0:
            #    continue
            labels.append(label)
            im_paths.append(anno_data[k][im_type])


        """
        for k in range(1100):
            im_paths.append(anno_data[str(k)][im_type])
            labels.append(anno_data[str(k)]["label"])
        """
        self.im_paths = im_paths
        self.labels = labels

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        # im = Image.open(self.im_paths[index])
        im = io.imread(self.im_paths[index])
        im = random_noise(im, mode='gaussian', var=0.002, seed=None, clip=True)
        im = np.array(im * 255, np.uint8)
        im = np.array(im, np.uint8)
        #print(self.im_paths[index].split("/")[-1].split(".")[0], "  \b")

        """
        im_name = self.im_paths[index].split("/")[-1].split(".")[0].split("_")[0]
        print(im_name)
        template_im = io.imread(str(RENDER_BG_SAVE_ALL / (im_name + ".jpg")))
        #print(np.max(im))
        template_im = np.array(template_im, np.uint8)

        histr1 = cv2.calcHist([im], [0], None, [256], [0, 256])
        histr2 = cv2.calcHist([template_im], [0], None, [256], [0, 256])

        plt.plot(histr1)
        plt.plot(histr2)
        plt.show()
        """
        #template_im = np.array(template_im * 255, np.uint8)
        #print(self.im_paths[index])


        im = Image.fromarray(im)
        # if self.mode == "val":
        #    im.show()
        #    cv2.waitKey(0)
        im = self.transform(im)
        label = self.labels[index]
        return (im/255.0, label) #torch.from_numpy(im).double().unsqueeze(dim=0), torch.from_numpy(label).long()

    def length(self):
        return len(self.im_paths)

    def __len__(self):
        return len(self.im_paths)

class CLassOnlyFeatureDataset(data.Dataset):

    def __init__(self, model, device, cropped_size=None, transform=[], rotation=False, noise=False, anno_path="", mode="train", im_type="PUF"):
        self.mode = mode
        self.cropped_size = cropped_size
        self.transform = transform
        self.rotation = rotation
        self.noise = noise
        self.im_type = im_type
        self.device = device
        self.phase = 1 # phase 1: training new pictures  phase 2: training all pictures

        anno_file = open(anno_path)
        anno_data = json.load(anno_file)
        im_paths = []
        labels = []

        for k in anno_data.keys():
            im_paths.append(anno_data[k][im_type])
            labels.append(anno_data[k]["label"])

        """
        for k in range(1100):
            im_paths.append(anno_data[str(k)][im_type])
            labels.append(anno_data[str(k)]["label"])
        """
        self.im_paths = im_paths
        self.features = []
        self.labels = labels
        self.model = model
        #angles_360 = np.arange(360).tolist()

        if self.mode == "train":
            angles = np.arange(180) * (360/180)
        elif self.mode == "add":
            angles = np.array((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330))
        else:
            angles = np.arange(180) * (360/180) + 1
            #angles = angles[::10]

        print("Mode: {}, with angles: {}".format(mode, angles))
        self.angles = angles

        #self.load_feature()
        #self.load_feature_from_file()

    def compute_features(self, ims):
        batch_size = 32
        batchs = len(ims) // batch_size
        features_list = []
        for i in range(batchs):
            im_batch = ims[(i*batch_size):((i+1)*batch_size)]
            features, _ = self.model(torch.cat(im_batch, 0))
            features = features.detach().cpu().numpy()
            for f in features:
                features_list.append(f)
        last_batch = ims[(batchs*batch_size):]
        last_features, _ = self.model(torch.cat(last_batch, 0))
        last_features = last_features.detach().cpu().numpy()
        for f in last_features:
            features_list.append(f)

        return features_list

    def load_feature_from_file(self, path):
        features = []
        with open(path/'features.csv', newline='') as csvfile:
            feature_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            for row in feature_reader:
                f = [float(x) for x in row]
                features.append(np.array(f))

        labels = []
        with open(path / 'labels.csv', newline='') as csvfile:
            label_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in label_reader:
                labels.append(int(row[0]))

        self.features = features
        self.labels = labels
        #if self.mode == "val":
        #    self.features = self.features[::10]
        #    self.labels = self.labels[::10]

    def load_feature(self):
        self.model.eval()
        features = []
        labels = []
        #self.test = []
        print("loading images")
        #all_images_tensors = []
        for im_p, label in zip(self.im_paths, self.labels):
            im_ori = io.imread(im_p)
            #print(im_p)

            if self.rotation:
                for rotate_angle in self.angles:
                    im = im_ori.copy()
                    if self.noise:
                        im = random_noise(im, mode='gaussian', var=0.002, seed=None, clip=True)
                    im = transform.rotate(im, rotate_angle, mode="edge")
                    im = np.array(im * 255, np.uint8)

                    im = Image.fromarray(im)
                    im = self.transform(im)
                    im = im/255.0
                    im = torch.unsqueeze(im, 0)
                    im = im.to(device=self.device, dtype=torch.float)
                    #all_images_tensors.append(im)
                    feature, _ = self.model(im)
                    features.append(feature.detach().cpu().numpy()[0])
                    labels.append(label)

                    #self.test.append([im_p, rotate_angle, label])
                    # if self.mode == "val":
                    #    im.show()
                    #    cv2.waitKey(0)
            elif self.noise:
                im = im_ori.copy()
                im = random_noise(im, mode='gaussian', var=0.002, seed=None, clip=True)
                im = np.array(im * 255, np.uint8)

                im = Image.fromarray(im)
                im = self.transform(im)
                im = im / 255.0
                im = torch.unsqueeze(im, 0)
                im = im.to(device=self.device, dtype=torch.float)
                feature, _ = self.model(im)
                features.append(feature.detach().cpu().numpy()[0])
                labels.append(label)
            else:
                im = im_ori.copy()
                im = Image.fromarray(im)
                im = self.transform(im)
                im = im / 255.0
                im = torch.unsqueeze(im, 0)
                im = im.to(device=self.device, dtype=torch.float)
                feature, _ = self.model(im)
                features.append(feature.detach().cpu().numpy()[0])
                labels.append(label)

        #print("compute fetures")
        self.features = features #self.compute_features(all_images_tensors)
        self.labels = labels

        #print(self.test)

    def add_feature(self, add_anno, index):
        self.added_features = []
        self.added_labels = []
        if index > len(self.im_paths)-1:
            self.model.eval()
            anno_file = open(add_anno)
            anno_data = json.load(anno_file)

            im_p = anno_data[str(index)][self.im_type]
            label = anno_data[str(index)]["label"]

            self.im_paths.append(im_p)

            im_ori = io.imread(im_p)
            print(im_p)

            print("adding feature")
            if self.rotation:
                for rotate_angle in self.angles:
                    im = im_ori.copy()
                    if self.noise:
                        im = random_noise(im, mode='gaussian', var=0.002, seed=None, clip=True)
                    im = transform.rotate(im, rotate_angle, mode="edge")
                    im = np.array(im * 255, np.uint8)

                    im = Image.fromarray(im)
                    im = self.transform(im)
                    im = im / 255.0
                    im = torch.unsqueeze(im, 0)
                    im = im.to(device=self.device, dtype=torch.float)
                    feature, _ = self.model(im)
                    self.features.append(feature.detach().cpu().numpy()[0])
                    self.labels.append(label)
                    self.added_features.append(feature.detach().cpu().numpy()[0])
                    self.added_labels.append(label)
                    # if self.mode == "val":
                    #    im.show()
                    #    cv2.waitKey(0)
            elif self.noise:
                im = im_ori.copy()
                im = random_noise(im, mode='gaussian', var=0.002, seed=None, clip=True)
                im = np.array(im * 255, np.uint8)

                im = Image.fromarray(im)
                im = self.transform(im)
                im = im / 255.0
                im = torch.unsqueeze(im, 0)
                im = im.to(device=self.device, dtype=torch.float)
                feature, _ = self.model(im)
                self.features.append(feature.detach().cpu().numpy()[0])
                self.labels.append(label)
                self.added_features.append(feature.detach().cpu().numpy()[0])
                self.added_labels.append(label)
            else:
                im = im_ori.copy()
                im = Image.fromarray(im)
                im = self.transform(im)
                im = im / 255.0
                im = torch.unsqueeze(im, 0)
                im = im.to(device=self.device, dtype=torch.float)
                feature, _ = self.model(im)
                self.features.append(feature.detach().cpu().numpy()[0])
                self.labels.append(label)
                self.added_features.append(feature.detach().cpu().numpy()[0])
                self.added_labels.append(label)

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        # im = Image.open(self.im_paths[index])
        if self.phase == 1:
            feature = self.added_features[index]
            feature = torch.from_numpy(feature)
            label = self.added_labels[index]
        else:
            feature = self.features[index]
            feature = torch.from_numpy(feature)
            label = self.labels[index]
        return (feature, label) #torch.from_numpy(im).double().unsqueeze(dim=0), torch.from_numpy(label).long()

    def length(self):
        return len(self.im_paths)

    def feature_num(self):
        return len(self.features)

    def added_feature_num(self):
        return len(self.angles)

    def __len__(self):
        if self.phase == 1:
            return len(self.added_features)
        else:
            return len(self.features)

class CLassOnlyDataset(data.Dataset):

    def __init__(self, cropped_size=None, transform=[], rotation=False, noise=False, anno_path="", mode="train", im_type="PUF"):
        self.mode = mode
        self.cropped_size = cropped_size
        self.transform = transform
        self.rotation = rotation
        self.noise = noise
        self.im_type = im_type

        anno_file = open(anno_path)
        anno_data = json.load(anno_file)
        im_paths = []
        labels = []

        for k in anno_data.keys():
            im_paths.append(anno_data[k][im_type])
            labels.append(anno_data[k]["label"])
        """
        for k in range(1100):
            im_paths.append(anno_data[str(k)][im_type])
            labels.append(anno_data[str(k)]["label"])
        """
        self.im_paths = im_paths
        self.labels = labels

        #angles_360 = np.arange(360).tolist()

        if self.mode == "train":
            angles = np.arange(180) * (360/180)
        elif self.mode == "add":
            angles = np.array((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330))
        else:
            angles = np.arange(180) * (360/180) + 1

        print("Mode: {}, with angles: {}".format(mode, angles))
        self.angles = angles

    def add_im(self, add_anno, index):
        anno_file = open(add_anno)
        anno_data = json.load(anno_file)

        im_p = anno_data[str(index)][self.im_type]
        label = anno_data[str(index)]["label"]

        self.im_paths.append(im_p)
        self.labels.append(label)

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        # im = Image.open(self.im_paths[index])
        im = io.imread(self.im_paths[index])
        noised = False

        if self.rotation:
            rotate_angle = random.choice(self.angles)
            im = transform.rotate(im, rotate_angle, mode="edge")

        if self.noise:
            scalar = 1 # random.randint(0, 3)/3.0 # add noise always
            if scalar > 0:
                im = random_noise(im, mode='gaussian', var=0.002*scalar, seed=None, clip=True)
                noised = True
            else:
                im=im

        if noised or self.rotation:
            im = np.array(im*255, np.uint8)
        #cv2.imshow("before", im)
        im = render_bg.processing(im)
        im = np.array(im, np.uint8)
        #cv2.imshow("after", im)
        #cv2.waitKey(0)
        im = Image.fromarray(im)
        #if self.mode == "val":
        #    im.show()
        #    cv2.waitKey(0)
        im = self.transform(im)
        label = self.labels[index]
        return (im/255.0, label) #torch.from_numpy(im).double().unsqueeze(dim=0), torch.from_numpy(label).long()

    def length(self):
        return len(self.im_paths)

    def __len__(self):
        return len(self.im_paths)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, input, target):
        angle = random.randint(self.angles[0], self.angles[1])
        return TF.rotate(input, angle), TF.rotate(target, angle)

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

def _croppad(img, kpt, center, w, h):
    num = len(kpt)
    height, width, _ = img.shape
    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    # calculate offset
    offset_up = -1 * (h / 2 - center[0])
    offset_left = -1 * (w / 2 - center[1])

    for i in range(num):
        kpt[i][0] -= offset_left
        kpt[i][1] -= offset_up

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height
    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

    return np.ascontiguousarray(new_img), kpt

def _get_keypoints(ann):
    kpt = np.zeros((len(ann) - 2, 3))
    for i in range(2, len(ann)):
        str = ann[i]
        [x_str, y_str, vis_str] = str.split('_')
        kpt[i - 2, 0], kpt[i - 2, 1], kpt[i - 2, 2] = int(x_str), int(y_str), int(vis_str)
    return kpt


def _generate_heatmap(img, kpt, stride, sigma):
    height, width, _ = img.shape
    heatmap = np.zeros((height / stride, width / stride, len(kpt) + 1), dtype=np.float32)  # (24 points + background)
    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == -1:  # not labeled
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        for h in range(height):
            for w in range(width):
                xx = start + w * stride
                yy = start + h * stride
                dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                if dis > 4.6052:
                    continue
                heatmap[h][w][i] += math.exp(-dis)
                if heatmap[h][w][i] > 1:
                    heatmap[h][w][i] = 1

    heatmap[:, :, -1] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)  # for background
    return heatmap

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        dataset_loader(cropped_size=220),
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)
    for i, (input, heatmap) in enumerate(train_loader):
        print(i)
