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

class feature_loader(data.Dataset):
    def __init__(self, number_class, feature_list=[], given_label=None):
        self.feature_list = feature_list
        self.given_label = given_label
        self.num_class = number_class

    def __getitem__(self, index):
        if self.given_label is None:
            feature, label = self.feature_list[index]
        else:
            feature = self.feature_list[index]
            label = self.given_label
        feature = torch.squeeze(feature, 0)
        label = np.array(label)
        #labels = np.zeros(self.num_class, np.long)
        #labels[label] = 1
        label = torch.from_numpy(label).long()

        return feature, label

    def __len__(self):
        return len(self.feature_list)


class im_feature_loader(data.Dataset):
    def __init__(self, number_class, feature_list=[]):
        self.feature_list = feature_list
        self.num_class = number_class

    def __getitem__(self, index):
        im, feature, label = self.feature_list[index]
        feature = torch.squeeze(feature, 0)
        label = np.array(label)
        #labels = np.zeros(self.num_class, np.long)
        #labels[label] = 1
        label = torch.from_numpy(label).long()

        return im, feature, label

    def __len__(self):
        return len(self.feature_list)

class im_loader(data.Dataset):
    def __init__(self, number_class, path, cropped_sie=512, random_rotate = True):
        self.path = path
        self.im_files = os.listdir(path)
        self.num_class = number_class
        self.rotate = random_rotate
        self.cropped_size = cropped_sie

    def __getitem__(self, index):
        file_name = self.im_files[index]
        im = io.imread(self.path + file_name)
        im = transform.resize(im, (self.cropped_size, self.cropped_size))
        if self.rotate:
            angle = np.random.randn(1)[0] * 360
            im = transform.rotate(im, angle)
        label = np.array(int(file_name[:-9]))
        #labels = np.zeros(self.num_class, np.long)
        #labels[label] = 1
        label = torch.from_numpy(label).long()
        im = np.expand_dims(im, axis=0)
        img = torch.from_numpy(im).double()
        return img, label

    def __len__(self):
        return len(self.im_files)

class dataset_loader(data.Dataset):

    def __init__(self, cropped_size, both_trans = None, img_path = "data/train/dataset/", binary_path = "data/train/dataset/", mode = "train"):
        self.im_file_path = img_path
        self.binary_file_path = binary_path
        self.cropped_size = cropped_size

        self.im_paths = os.listdir(self.im_file_path)

        self.both_transform = both_trans
        self.mode = mode
        self.indices = np.arange(len(self.im_paths))
        self.index = 0
        self.index_fixed = False

    def get_one_im(self, im_name):
        im = io.imread(os.path.join(self.im_file_path, im_name))
        im = transform.resize(im, (self.cropped_size, self.cropped_size))

        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        img = torch.from_numpy(im).double()

        return img

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        rotate_angle = np.random.randn(1)[0] * 360
        #print(self.index_fixed)
        if self.index_fixed:
            #print(index, self.index)
            if index == self.index:
                self.index = np.random.choice(np.where(self.indices != self.index)[0])
            im_path_pos = self.im_paths[self.index]
            im_name_pos = im_path_pos[:-4]
            # print(self.im_file_path + im_path)
            im_pos = io.imread(os.path.join(self.im_file_path, im_path_pos))

            # binary = io.imread(os.path.join(self.binary_file_path, im_name + "_otsu.jpg"))

            # anno_im = cv2.erode(anno_im, (3, 3), iterations=2)

            if self.both_transform is not None:
                if self.mode == "train":
                    im_pos = transform.rotate(im_pos, rotate_angle)
                    # binary = transform.rotate(binary, rotate_angle)
            im_pos = transform.resize(im_pos, (self.cropped_size, self.cropped_size))
            # binary = transform.resize(binary, (self.cropped_size, self.cropped_size))
        else:
            im_pos = None
        rotate_angle = np.random.randn(1)[0] * 360
        im_path = self.im_paths[index]
        im_name = im_path[:-4]
        #print(self.im_file_path + im_path)
        im = io.imread(os.path.join(self.im_file_path, im_path))

        #binary = io.imread(os.path.join(self.binary_file_path, im_name + "_otsu.jpg"))

        #anno_im = cv2.erode(anno_im, (3, 3), iterations=2)

        if self.both_transform is not None:
            if self.mode == "train":
                im = transform.rotate(im, rotate_angle)
                #binary = transform.rotate(binary, rotate_angle)
        im = transform.resize(im, (self.cropped_size, self.cropped_size))
        #binary = transform.resize(binary, (self.cropped_size, self.cropped_size))

        #fig, (ax0, ax1) = plt.subplots(1, 2)
        # ax0.imshow(gray, cmap='gray')
        #ax0.imshow(im, cmap='gray')
        #ax1.imshow(binary, cmap='gray')

        #plt.show()
        ims = np.zeros((2, 1, self.cropped_size, self.cropped_size))
        #im = np.expand_dims(im, axis=0)
        if im_pos is not None:
            ims[0, :, :, :] = im_pos
            ims[1, :, :, :] = im
            img = torch.from_numpy(ims).double()
            labels = np.zeros(2, np.long)
            labels[0] = (im_name_pos[0] == "t") * 1
            labels[1] = (im_name[0] == "t") * 1
            ori_name = [im_name_pos, im_name]

            label = torch.from_numpy(labels).long()
        else:
            ims[0, :, :, :] = im
            ims[1, :, :, :] = im
            img = torch.from_numpy(ims).double()
            labels = np.zeros(2, np.long)
            labels[0] = (im_name[0] == "t") * 1
            labels[1] = (im_name[0] == "t") * 1
            ori_name = [im_name, im_name]

            label = torch.from_numpy(labels).long()
        #binary = np.expand_dims(binary, axis=0)
        #binary = torch.from_numpy(binary).double()

        return index, ori_name, img, label #binary

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

class AEDataset(data.Dataset):

    def __init__(self, cropped_size=None, trans=[], anno_path="", mode="train", im_type="PUF"):
        self.mode = mode
        self.cropped_size = cropped_size
        self.transformations = trans

        anno_file = open(anno_path)
        anno_data = json.load(anno_file)
        im_paths = []
        for k in anno_data.keys():
            im_paths.append(anno_data[k][im_type])

        self.im_paths = im_paths#[:16]

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        im = io.imread(self.im_paths[index])
        #print("ori", np.max(im))
        if self.transformations == "None":
            im = im / 255.0 #(im / 255.0 - 0.5) * 2
        else:
            if "rotation" in self.transformations:
                rotate_angle = np.random.randn(1)[0] * 360
                im = transform.rotate(im, rotate_angle, mode="edge")
                #print("rotate", np.max(im))

            if "brightness" in self.transformations:
                im = brightness(im)

            if "noise" in self.transformations:
                gauss = np.random.normal(0, 0.04, (im.shape[0], im.shape[1]))
                im += gauss

            if "random":
                random_flag = np.random.randint(4, size=1)[0]
                if random_flag == 2:
                    im = np.random.rand(im.shape[0], im.shape[1])
                    #im = (im - 0.5) * 2

            #im = (im - 0.5)*2

        if self.cropped_size is not None:
            im = transform.resize(im, (self.cropped_size, self.cropped_size))

        return torch.from_numpy(im).double().unsqueeze(dim=0)

    def length(self):
        return len(self.im_paths)

    def __len__(self):
        return len(self.im_paths)

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

class JEMDataset(data.Dataset):

    def __init__(self, cropped_size=None, transform=[], rotation=False, anno_path="", mode="train", im_type="PUF"):
        self.mode = mode
        self.cropped_size = cropped_size
        self.transform = transform
        self.rotation = rotation

        anno_file = open(anno_path)
        anno_data = json.load(anno_file)
        im_paths = []
        labels = []
        """
        for k in anno_data.keys():
            im_paths.append(anno_data[k][im_type])
            labels.append(anno_data[k]["label"])
        """
        for k in range(1980):
            im_paths.append(anno_data[str(k)][im_type])
            labels.append(anno_data[str(k)]["label"])

        self.im_paths = im_paths#[:10]
        self.labels = labels#[:10]

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        # im = Image.open(self.im_paths[index])
        im = io.imread(self.im_paths[index])
        if self.rotation:
            rotate_angle = np.random.randn(1)[0] * 360
            im = transform.rotate(im, rotate_angle, mode="edge")
            im = np.array(im*255, np.uint8)
        im = Image.fromarray(im)
        #im.show()
        #print("ori", np.max(im))
        """
        if self.transformations == "None":
            im = im #(im / 255.0 - 0.5) * 2
        else:
            if "rotation" in self.transformations:
                rotate_angle = np.random.randn(1)[0] * 360
                im = transform.rotate(im, rotate_angle, mode="edge")
                #print("rotate", np.max(im))

            if "brightness" in self.transformations:
                im = brightness(im)

            if "noise" in self.transformations:
                gauss = np.random.normal(0, 0.04, (im.shape[0], im.shape[1]))
                im += gauss

            if "random":
                random_flag = np.random.randint(4, size=1)[0]
                if random_flag == 2:
                    im = np.random.rand(im.shape[0], im.shape[1])
                    #im = (im - 0.5) * 2

            im = (im - 0.5)*2
        print(np.max(im))
        if self.cropped_size is not None:
            im = transform.resize(im, (self.cropped_size, self.cropped_size))
        print(np.max(im))
        """
        im = self.transform(im)
        label = self.labels[index]
        return (im, label) #torch.from_numpy(im).double().unsqueeze(dim=0), torch.from_numpy(label).long()

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
