from __future__ import print_function, division
import os
import argparse
import time
import pickle
import csv
from torch.utils.data import DataLoader
import numpy as np
import torchvision as tv, torchvision.transforms as tr
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from ebm import EBM
#import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc
from Dataloader import MyRotationTransform, CLassOnlyTestDataset
from classification import Resnet50 as Classification #, CNNclassifier
from config import *
from SimISSM import rmse, ssim, issm, fsim, fsim_torch
from skimage import io, draw, color, transform, exposure
from PixelWiseSim import Pixel_Wise_Sim, Pixel_Wise_Sim_360, iterated_fsim, fsim_360
from util import circle_area_otsu

modes = ['train', 'val']
import cv2

criterion = nn.CrossEntropyLoss()
sm = nn.Softmax(dim=1)

# %%Main
if __name__ == "__main__":

    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')

    parser.add_argument('--im_size', type=int, default=448, help='the size of image')
    parser.add_argument('--load_model', help='checkpoint you want to load for discriminator')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--num_classes', default=1300, type=int, help='number of classes')
    parser.add_argument('--top_k', default=5, type=int, help='top k test')
    args = parser.parse_args()
    main_path = "./" #os.path.dirname(os.path.realpath(__file__))

    print("Pytorch Version:", torch.__version__)
    print(args)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # train_root = os.path.join(main_path, args.train_set, "im")
    test_root = TEST_PATH / "test_smaller.json"  # os.path.join(main_path, args.train_set, "binary")
    #test_root = TRAIN_PATH / "train.json"


    transform_train = tr.Compose(
        [tr.Resize((args.im_size, args.im_size), interpolation=0),
         tr.ToTensor(),]
    )

    # Initialize networks
    dataset = CLassOnlyTestDataset(first_classes=args.num_classes,
                                   anno_path=test_root,
                                   transform=transform_train,
                                   im_type="PUF_render_bg")
    if batch_size > dataset.length():




        loader = DataLoader(dataset, batch_size=dataset.length(), shuffle=True,
                                  drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    #classNet = CNNclassifier(num_class=nu_classes).to(device).float()
    classNet = Classification(num_class=args.num_classes).to(device).float()

    # use multiple gpus
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print("Let's use", n_gpus, "GPUs!")
        batch_size = n_gpus * args.batch_size
        #disNet = nn.DataParallel(CNNclassifier(num_class=nu_classes).to(device))
        disNet = nn.DataParallel(Classification(num_class=args.num_classes).to(device))

    # load models if requested
    if args.load_model is not None:
        print("Ae model loaded: " + args.load_model)
        classNet.load_state_dict(torch.load(args.load_model))

    #classNet.add_classes(args.num_classes, args.num_classes+1)
    classNet.to(device)
    classNet.eval()
    print(classNet)

    correct_pred = 0
    total_pred = 0
    for i, data in enumerate(loader, 0):
        start_time = time.time()
        im_batch, labels = data  # next(iter(loader[mode]))
        # print(img_name)
        im_batch = im_batch.to(device=device, dtype=torch.float)
        labels = labels.to(device=device, dtype=torch.long)

        predictions = classNet(im_batch)
        outputs = sm(predictions)
        probs, predictions = torch.max(outputs, 1)
        probs_k, predictions_k = torch.topk(outputs, args.top_k)
        print("used time", time.time() - start_time)

        for input_im, label, prediction, prediction_k, p in zip(im_batch.detach().cpu().numpy(),
                                                                 labels.detach().cpu().numpy(),
                                                                 predictions.detach().cpu().numpy(),
                                                                 predictions_k.detach().cpu().numpy(),
                                                                 probs.detach().cpu().numpy()):
            """
            data_base_im = io.imread(str(RENDER_BG_SAVE_ALL / (str(prediction) + ".jpg")))
            data_base_im = transform.resize(data_base_im, (448, 448))
            #print(str(RENDER_BG_SAVE_ALL / (str(prediction) + "jpg")))
            #cv2.imshow("show", data_base_im)
            #cv2.waitKey(0)
            data_base_im = np.array(data_base_im*255.0, np.uint8)
            input_im = np.array(input_im*255.0, np.uint8)
            #data_base_im = np.expand_dims(data_base_im, axis=0)
            input_im = input_im[0, :, :]
            data_base_im = cv2.cvtColor(data_base_im, cv2.COLOR_GRAY2RGB) #np.expand_dims(data_base_im, axis=2)
            input_im = cv2.cvtColor(input_im, cv2.COLOR_GRAY2RGB) # np.expand_dims(input_im, axis=2)
            print(data_base_im.shape, input_im.shape)
            sim = fsim_torch(data_base_im, input_im)
            """
            # print(label, prediction, p)
            # if p < 0.7:
            #    correct_pred += 1
            #if label > 1599:
            #    continue
            #if label == -1 or p<0.8:
            #    continue
            if label >= args.num_classes:
                continue

            sims = []

            #input_im[np.where(input_im>0)]=1
            input_im = input_im[0, :, :]

            #_, _, input_im = circle_area_otsu(input_im)
            #input_im = np.array(input_im * 255.0, np.uint8)
            #cv2.imshow("show", input_im)
            #cv2.waitKey(0)
            for p_k in prediction_k:
                data_base_im = io.imread(str(RENDER_BG_SAVE_ALL / (str(p_k) + ".jpg")))
                data_base_im = transform.resize(data_base_im, (448, 448))
                #data_base_im = np.array(data_base_im * 255.0, np.uint8)
                #cv2.imshow("show_ori", data_base_im)
                #cv2.waitKey(0)
                # print(str(RENDER_BG_SAVE_ALL / (str(prediction) + "jpg")))
                # cv2.imshow("show", data_base_im)
                # cv2.waitKey(0)
                #data_base_im = np.array(data_base_im * 255.0, np.int)
                #data_base_im[np.where(data_base_im > 0)] = 255
                _, similarity = fsim_360(im=input_im, target=data_base_im, angle_thre=5)
                sims.append(similarity)
            sim_ind = np.argmax(sims)
            #prediction = predictions_k[sim_ind]
            """
            
            for t_im, label, predictions, ps in zip(t_ims, t_labels, predictions_k, probs_k):
                # print(predictions, ps)
                sims = []
                for prediction in predictions:
                    _, similarity = Pixel_Wise_Sim(im=t_im, target=pre_ims_list[prediction - 1], angle_thre=1)
                    sims.append(similarity)
                sim_ind = np.argmax(sims)
                test_result_csv_writer.writerow([d_i,
                                                 label.cpu().detach().numpy(),
                                                 predictions.cpu().detach().numpy()[sim_ind],
                                                 sims[sim_ind],
                                                 ps.cpu().detach().numpy()[sim_ind]])
                d_i += 1
            """

            if prediction >= args.num_classes:
                prediction = -1
            with open(RESULT_PATH / "test_result_new_fsim_tesst.csv", "a+", newline='') as f:
                valwriter = csv.writer(f, delimiter=';',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
                valwriter.writerow([label, prediction, prediction_k.tolist(), p, sims])

            print(label, prediction, prediction_k, p, sims)
            if label == prediction:
                if p>0.7:
                    correct_pred += 1
            total_pred += 1
        #if i > 10:
        #    break
    accuracy = (correct_pred * 100.0) / total_pred
    print("train accuracy is: %", accuracy)


