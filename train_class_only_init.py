from __future__ import print_function, division
import os
import argparse
import time
import pickle
import csv
from torch.utils.data import DataLoader
import torchvision as tv, torchvision.transforms as tr
import torch
import torch.optim as optim
#import nibabel as nib
import torch.nn as nn
from Dataloader import MyRotationTransform, CLassOnlyDataset
from classification import Resnet50 as Classification #, CNNclassifier
from config import *
modes = ['train', 'val']

criterion = nn.CrossEntropyLoss()
sm = nn.Softmax(dim=1)

# %%Main
if __name__ == "__main__":

    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')


    parser.add_argument('--num_epochs', default=3001, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.0, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.9999, help='learning rate decay')
    parser.add_argument('--im_size', type=int, default=448, help='the size of image')
    parser.add_argument('--save_model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--train_set', default='data/train', help='name of dataset path')
    parser.add_argument('--val_set', default='data/val', help='name of valset path')
    parser.add_argument('--experiment', default='train', help='name of experiment')
    parser.add_argument('--load_model', help='checkpoint you want to load for discriminator')
    parser.add_argument('--epoch_start', default=0, type=int, help='epoch you want to start from')
    parser.add_argument('--save_visu', default=True, help='saves on training/valing visualization')
    parser.add_argument('--num_channels', default=8, type=int, help='Number of Channels for the CNN')
    parser.add_argument('--num_latents', default=4096, type=int, help='dimension of latent space')
    parser.add_argument('--print_epoch', default=10, type=int, help='dimension of latent space')
    parser.add_argument('--save_epoch', default=100, type=int, help='dimension of latent space')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    args = parser.parse_args()
    main_path = "./" #os.path.dirname(os.path.realpath(__file__))

    print("Pytorch Version:", torch.__version__)
    print("Experiment: " + args.experiment)
    print(args)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # train_root = os.path.join(main_path, args.train_set, "im")
    train_root = TRAIN_PATH / "train.json"  # os.path.join(main_path, args.train_set, "binary")
    # val_root = os.path.join(main_path, args.val_set, "im")
    val_root = TRAIN_PATH / "train.json"  # VAL_PATH / "val.json" #os.path.join(main_path, args.val_set, "binary")

    # create variables as dictionaries
    dataset = {}
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
            dataset[mode] = CLassOnlyDataset(anno_path=train_root,
                                       transform=transform_train,
                                       mode="add",
                                       rotation=True,
                                       noise=True,
                                       im_type="PUF")
            if batch_size > dataset[mode].length():
                loader[mode] = DataLoader(dataset[mode], batch_size=dataset[mode].length(), shuffle=True,
                                          drop_last=True)
            else:
                loader[mode] = DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)

        else:
            dataset[mode] = CLassOnlyDataset(anno_path=val_root,
                                       transform=transform_train,
                                       mode=mode,
                                       rotation=True,
                                       noise=True,
                                       im_type="PUF")
            if batch_size > dataset[mode].length():
                loader[mode] = DataLoader(dataset[mode], batch_size=dataset[mode].length(), shuffle=True,
                                          drop_last=True)
            else:
                loader[mode] = DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)

        loss_dic[mode] = []
        cdiv_dic[mode] = []
        reg_dic[mode] = []

    num_classes = dataset["train"].length()
    #classNet = CNNclassifier(num_class=nu_classes).to(device).float()
    classNet = Classification(num_class=num_classes).to(device).float()

    # use multiple gpus
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print("Let's use", n_gpus, "GPUs!")
        batch_size = n_gpus * args.batch_size
        #disNet = nn.DataParallel(CNNclassifier(num_class=nu_classes).to(device))
        disNet = nn.DataParallel(Classification(num_class=num_classes).to(device))

    # load models if requested
    if args.load_model is not None:
        print("Ae model loaded: " + args.load_model)
        classNet.load_state_dict(torch.load(args.load_model))

    print(classNet)

    # optimizesr
    optimizer = optim.Adam(classNet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.999)  # Exponential decay over epochs



    # Start Training/valing
    start_time = time.time()
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
        for i, data in enumerate(loader[mode], 0):
            torch.cuda.empty_cache()

            # print(torch.cuda.get_device_properties(0).total_memory/1000000)
            # print(torch.cuda.memory_cached(0)/1000000)
            # print(torch.cuda.memory_allocated(0)/1000000)

            imgs, labels = data  # next(iter(loader[mode]))
            # print(img_name)
            imgs = imgs.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)


            pre = classNet(imgs)

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

        if mode == 'train':
            if (epoch+1) % args.print_epoch == 0 or epoch == 1:

                correct_pred = 0
                total_pred = 0
                for i, data in enumerate(loader[mode], 0):

                    im_batch, labels = data  # next(iter(loader[mode]))
                    # print(img_name)
                    im_batch = im_batch.to(device=device, dtype=torch.float)
                    labels = labels.to(device=device, dtype=torch.long)

                    predictions = classNet(im_batch)
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
                accuracy = (correct_pred * 100.0) / total_pred
                print("train accuracy is: %", accuracy)
                with open(MODEL_SAVE_CLASS_INIT / "train_epoch_loss.csv", "a+", newline='') as f:
                    trainwriter = csv.writer(f, delimiter=' ',
                                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    trainwriter.writerow([epoch, mode, avg_loss, accuracy])
                    #f.write("Epoch {}: {}    loss: {}  acc: {} %  \n".format(epoch, mode, avg_loss, accuracy))
                    #print("\n")
        # Saving current model
        if mode == 'val':
            if epoch%args.print_epoch == 0:

                correct_pred = 0
                total_pred = 0
                for i, data in enumerate(loader[mode], 0):

                    im_batch, labels = data  # next(iter(loader[mode]))
                    # print(img_name)
                    im_batch = im_batch.to(device=device, dtype=torch.float)
                    labels = labels.to(device=device, dtype=torch.long)

                    predictions = classNet(im_batch)
                    outputs = sm(predictions)
                    probs, predictions = torch.max(outputs, 1)

                    for label, prediction, p in zip(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), probs.detach().cpu().numpy()):
                        #print(label, prediction, p)
                        # if p < 0.7:
                        #    correct_pred += 1
                        if label == prediction:
                            correct_pred += 1
                        total_pred += 1
                accuracy = (correct_pred * 100.0) / total_pred
                print("val accuracy is: %", accuracy)
                with open(MODEL_SAVE_CLASS_INIT / "val_epoch_loss.csv", "a+", newline='') as f:
                    valwriter = csv.writer(f, delimiter=';',
                                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    valwriter.writerow([epoch, mode, avg_loss, accuracy])
                    #f.write("Epoch {}: {}   loss: {}    acc: {} %  \n".format(epoch, mode, avg_loss, accuracy))
                    #print("\n")
        if epoch%args.save_epoch == 0:
            torch.save(classNet.state_dict(), MODEL_SAVE_CLASS_INIT / ('class_init_' + str(epoch) + '.pth'))
            with open(MODEL_SAVE_CLASS_INIT / (args.experiment + ".p"), 'wb') as f:
                pickle.dump([loss_dic], f)
            print("--- %s seconds ---" % (time.time() - start_time))
