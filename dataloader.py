import torch.utils.data as data
import random
import json
from skimage import io, draw, color, transform, exposure
from skimage.util import random_noise
import numpy as np
from PIL import Image

class CLassOnlyDataset(data.Dataset):

    def __init__(self, cropped_size=None, transform=[], rotation=False, noise=False, anno_path="", mode="train", im_type="PUF"):
        self.mode = mode
        self.cropped_size = cropped_size
        self.transform = transform
        self.rotation = rotation
        self.noise = noise

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
        else:
            angles = np.arange(180) * (360/180) + 1

        print("Mode: {}, with angles: {}", mode, angles)
        self.angles = angles

    def __getitem__(self, index):
        # ---------------- read info -----------------------
        # get the positive and negative image
        # im = Image.open(self.im_paths[index])
        im = io.imread(self.im_paths[index])
        if self.noise:
            im = random_noise(im, mode='gaussian', seed=None, clip=True)
        if self.rotation:
            rotate_angle = random.choice(self.angles)
            im = transform.rotate(im, rotate_angle, mode="edge")
        if self.noise or self.rotation:
            im = np.array(im*255, np.uint8)

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