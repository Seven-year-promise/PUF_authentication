from __future__ import print_function, division
import os
import argparse
import time
import csv
import pickle
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
from Dataloader import MyRotationTransform, CLassOnlyFeatureDataset
from classification import Resnet50 as Classification #, CNNclassifier
from config import *
modes = ['train', 'val']

import cv2

criterion = nn.CrossEntropyLoss()
sm = nn.Softmax(dim=1)

# %%Main
if __name__ == "__main__":

    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')


    parser.add_argument('--num_epochs', default=5001, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.0, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.9999, help='learning rate decay')
    parser.add_argument('--acc_thre', type=float, default=96, help='the threshold of val accuracy to stop training (%)')
    parser.add_argument('--im_size', type=int, default=448, help='the size of image')
    parser.add_argument('--save_model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--train_set', default='data/train', help='name of dataset path')
    parser.add_argument('--val_set', default='data/val', help='name of valset path')
    parser.add_argument('--experiment', default='train', help='name of experiment')
    parser.add_argument('--load_model', default='results/class_init/class_init_2250.pth', help='checkpoint you want to load for discriminator')
    parser.add_argument('--epoch_start', default=0, type=int, help='epoch you want to start from')
    parser.add_argument('--save_visu', default=True, help='saves on training/valing visualization')
    parser.add_argument('--num_channels', default=8, type=int, help='Number of Channels for the CNN')
    parser.add_argument('--pre_num_classes', default=1100, type=int, help='Number of classes in the previous training')
    parser.add_argument('--a_n_classes', default=3, type=int, help='Number of added classes to the previous training')
    parser.add_argument('--num_latents', default=4096, type=int, help='dimension of latent space')
    parser.add_argument('--print_epoch', default=5, type=int, help='dimension of latent space')
    parser.add_argument('--save_epoch', default=5, type=int, help='dimension of latent space')
    parser.add_argument('--batch_size', default=6, type=int, help='Batch Size')
    parser.add_argument('--return_hidden', default=True, type=bool, help='Whether return the feature vector')


    args = parser.parse_args()
    main_path = "./" #os.path.dirname(os.path.realpath(__file__))

    with open(MODEL_SAVE_CLASS_ADD / "hyper_para.txt", "w") as output:  ## creates new file but empty
        for arg in vars(args):
            print(arg, getattr(args, arg))
            output.write(str(arg) + "\t" + str(getattr(args, arg)) + "\n")

    print("Pytorch Version:", torch.__version__)
    print("Experiment: " + args.experiment)
    print(args)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # initialize the previous model
    classNet = Classification(num_class=args.pre_num_classes, return_hidden=args.return_hidden).to(device).float()

    # load models if requested
    if args.load_model is not None:
        print("Ae model loaded: " + args.load_model)
        classNet.load_state_dict(torch.load(args.load_model))

    for param in classNet.parameters():
        param.requires_grad = False

    dataset = {}
    train_root = TRAIN_PATH / "train.json"  # os.path.join(main_path, args.train_set, "binary")
    # val_root = os.path.join(main_path, args.val_set, "im")
    val_root = TRAIN_PATH / "train.json"

    feature_train_path = INIT_FEATURE_PATH / "train/"
    feature_val_path = INIT_FEATURE_PATH / "val/"

    transform_train = tr.Compose(
        [tr.Resize((args.im_size, args.im_size), interpolation=0),
         tr.ToTensor(), ]
    )

    for i, mode in enumerate(modes):
        if mode == 'train':
            dataset[mode] = CLassOnlyFeatureDataset(model=classNet,
                                                    device=device,
                                                    anno_path=train_root,
                                                    transform=transform_train,
                                                    mode="add",
                                                    rotation=True,
                                                    noise=True,
                                                    im_type="PUF_render_bg")
            dataset[mode].load_feature_from_file(feature_train_path)
        else:
            dataset[mode] = CLassOnlyFeatureDataset(model=classNet,
                                                    device=device,
                                                    anno_path=val_root,
                                                    transform=transform_train,
                                                    mode=mode,
                                                    rotation=True,
                                                    noise=True,
                                                    im_type="PUF_render_bg")
            dataset[mode].load_feature_from_file(feature_val_path)
    # begin to add new pufs and update the model
    for a_n in range(1, args.a_n_classes+1):
        val_accuracy = 0
        train_accuracy = 0
        # train_root = os.path.join(main_path, args.train_set, "im")
        add_root = ADD_PATH / ("add_" + str(a_n) + ".json")

        model_save_path = MODEL_SAVE_CLASS_ADD / ("add_" + str(a_n))

        model_save_path.mkdir(parents=True, exist_ok=True)

        # create variables as dictionaries

        loader = {}
        loss_dic = {}
        cdiv_dic = {}
        reg_dic = {}

        transform_train = tr.Compose(
            [tr.Resize((args.im_size, args.im_size), interpolation=0),
             tr.ToTensor(),]
        )

        # Initialize networks
        # aeNet = Autoencoder(args.num_channels, args.num_latents, args.num_class).to(device)  # Since the images are b&w, they initially have 1 ch
        for i, mode in enumerate(modes):
            if mode == 'train':
                dataset[mode].add_feature(add_anno=add_root, index=args.pre_num_classes+a_n-1)
                dataset[mode].phase = 2
                if batch_size > dataset[mode].feature_num():
                    loader[mode] = DataLoader(dataset[mode], batch_size=dataset[mode].feature_num(), shuffle=True,
                                              drop_last=False)
                else:
                    loader[mode] = DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=False)

            else:
                dataset[mode].add_feature(add_anno=add_root, index=args.pre_num_classes+a_n-1)
                dataset[mode].phase = 2
                if batch_size > dataset[mode].feature_num():
                    loader[mode] = DataLoader(dataset[mode], batch_size=dataset[mode].feature_num(), shuffle=True,
                                              drop_last=False)
                else:
                    loader[mode] = DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=False)

            loss_dic[mode] = []
            cdiv_dic[mode] = []
            reg_dic[mode] = []

        num_classes = dataset["train"].length()

        if a_n > 0:
            classNet.add_classes(args.pre_num_classes+a_n-1, num_classes)
        else:
            classNet.cnn.classifier.weight.requires_grad = True
            classNet.cnn.classifier.bias.requires_grad = True
        classNet.to(device)

        print(classNet)

        # use multiple gpus
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print("Let's use", n_gpus, "GPUs!")
            batch_size = n_gpus * args.batch_size
            # disNet = nn.DataParallel(CNNclassifier(num_class=nu_classes).to(device))
            disNet = nn.DataParallel(classNet.to(device))

        # optimizesr
        optimizer = optim.Adam(classNet.cnn.classifier.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.999)  # Exponential decay over epochs

        # Start Training/valing

        for epoch in range(args.epoch_start+1, args.epoch_start + args.num_epochs+1):
            epoch_loss = {}
            epoch_cdiv_loss = {}
            epoch_reg_loss = {}
            if epoch%args.print_epoch == 0:
                mode = "val"
            else:
                mode = "train"

            if mode == 'train':
                classNet.train()
            else:
                classNet.eval()

            epoch_loss[mode] = []
            epoch_cdiv_loss[mode] = []
            epoch_reg_loss[mode] = []
            start_time = time.time()
            for i, data in enumerate(loader[mode], 0):
                torch.cuda.empty_cache()

                # print(torch.cuda.get_device_properties(0).total_memory/1000000)
                # print(torch.cuda.memory_cached(0)/1000000)
                # print(torch.cuda.memory_allocated(0)/1000000)

                features, labels = data  # next(iter(loader[mode]))
                # print(img_name)
                features = features.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)


                pre = classNet.cnn.classifier(features)

                loss = criterion(pre, labels)

                if mode == 'train':
                    optimizer.zero_grad()

                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                """
                print('[%d/%d][%d/%d]\tae_l: %.4f\tclass_loss: %.4f\tclass_lossProc: %.4f\n' % (
                epoch, args.num_epochs, i, len(loader[mode]), ae_loss.item(), ae_dice, ae_dice_proc))
                """
                # print('aeLoss: {:.4f}   kld loss: {}'.format(ae_loss, kld_loss))
                epoch_loss[mode].append(loss.item())

            if mode == 'train':
                scheduler.step()
            loss_dic[mode].append(epoch_loss[mode])

            avg_loss = sum(epoch_loss[mode]) / len(epoch_loss[mode])

            print('{} Epoch: {} \t  Loss: {:.4f}   Learning rate: {}'.format(mode,
                                                                               epoch,
                                                                               avg_loss,
                                                                               scheduler.get_last_lr()))

            # if args.save_visu and (epoch%50==0):
            print("--- for one epoch: %s seconds ---" % (time.time() - start_time))
            if mode == 'train':
                if (epoch+1) % args.print_epoch == 0 or epoch == 1:

                    correct_pred = 0
                    total_pred = 0
                    for i, data in enumerate(loader[mode], 0):

                        features_batch, labels = data  # next(iter(loader[mode]))
                        # print(img_name)
                        features_batch = features_batch.to(device=device, dtype=torch.float)
                        labels = labels.to(device=device, dtype=torch.long)

                        predictions = classNet.cnn.classifier(features_batch)
                        outputs = sm(predictions)
                        probs, predictions = torch.max(outputs, 1)

                        for label, prediction, p in zip(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                                                        probs.detach().cpu().numpy()):
                            # print(label, prediction, p)
                            # if p < 0.7:
                            #    correct_pred += 1
                            if label == prediction:
                                correct_pred += 1
                            total_pred += 1
                    train_accuracy = (correct_pred * 100.0) / total_pred
                    print("train accuracy is: %", train_accuracy, "in total:", total_pred)
                    with open(model_save_path / "train_epoch_loss.csv", "a+", newline='') as f:
                        trainwriter = csv.writer(f, delimiter=' ',
                                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        trainwriter.writerow([epoch, mode, avg_loss, train_accuracy])
                        #f.write("Epoch {}: {}    loss: {}  acc: {} %  \n".format(epoch, mode, avg_loss, accuracy))
                        #print("\n")
            # Saving current model
            if mode == 'val':
                if epoch%args.print_epoch == 0:

                    correct_pred = 0
                    total_pred = 0
                    for i, data in enumerate(loader[mode], 0):

                        features_batch, labels = data  # next(iter(loader[mode]))
                        # print(img_name)
                        features_batch = features_batch.to(device=device, dtype=torch.float)
                        labels = labels.to(device=device, dtype=torch.long)

                        predictions = classNet.cnn.classifier(features_batch)
                        outputs = sm(predictions)
                        probs, predictions = torch.max(outputs, 1)

                        for label, prediction, p in zip(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), probs.detach().cpu().numpy()):
                            #print(label, prediction, p)
                            # if p < 0.7:
                            #    correct_pred += 1
                            #print(label, prediction)
                            if label == prediction:
                                correct_pred += 1
                            total_pred += 1
                    val_accuracy = (correct_pred * 100.0) / total_pred
                    print("val accuracy is: %", val_accuracy, "in total:", total_pred)
                    with open(model_save_path / "val_epoch_loss.csv", "a+", newline='') as f:
                        valwriter = csv.writer(f, delimiter=' ',
                                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        valwriter.writerow([epoch, mode, avg_loss, val_accuracy])
                        #f.write("Epoch {}: {}   loss: {}    acc: {} %  \n".format(epoch, mode, avg_loss, accuracy))
                        #print("\n")
            if epoch%args.save_epoch == 0:
                torch.save(classNet.state_dict(), model_save_path / (str(epoch) + ".pth"))
                with open(model_save_path / (args.experiment + ".p"), 'wb') as f:
                    pickle.dump([loss_dic], f)
                print("--- %s seconds ---" % (time.time() - start_time))

            #if train_accuracy >= args.acc_thre:
            """
            if dataset["train"].phase == 1: # change to phase 2 after epoch 1
                val_accuracy = 0
                train_accuracy = 0
                print("exchange to phase 2")
                for m in modes:
                    dataset[m].phase = 2
                    if batch_size > dataset[m].feature_num():
                        loader[m] = DataLoader(dataset[m], batch_size=dataset[m].feature_num(), shuffle=True,
                                                  drop_last=False)
                    else:
                        loader[m] = DataLoader(dataset[m], batch_size=batch_size, shuffle=True, drop_last=False)
            """
            if train_accuracy >= args.acc_thre and val_accuracy >= args.acc_thre:
                with open(MODEL_SAVE_CLASS_ADD / "trained_opech_add.csv", "a+", newline='') as f:
                    valwriter = csv.writer(f, delimiter=' ',
                                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    valwriter.writerow([a_n, epoch])
                break