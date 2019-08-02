import os,sys
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import helper
#import simulation
from tqdm import tqdm
import time
import copy
import numpy as np
import torch
import metrics
import cv2
from metrics import MetricsCollection
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
#from unet_center_data_loader import dataloaders


def train_model(model, optimizer, lr_scheduler, data_loader,device, max_epochs=100): #loaders
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    has_waited = 0
    stop_training = False
    earlystop_patience = 5

    epoch_metrics = MetricsCollection()

    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
        #    if phase == 'train':
        #        model.train()  # Set model to training mode
        #    else:
        #        model.eval()  # Set model to evaluate mode

            batch_metrics = MetricsCollection()

            #########################
            # loader = loaders[phase]
            pbar = tqdm(data_loader, total=len(data_loader), desc="Epoch {} {}".format(epoch, phase), ncols=0)
            #for iter_id, batch in enumerate(pbar): #data_loader
            for ind, (img_id, batch) in enumerate(data_loader): #data_loader


                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=device, non_blocking=True)
                inputs,class_masks,center_masks,width_masks,height_masks =batch['input'],batch['gt_segmap'], batch['gt_ctmap'], batch['gt_widmap'], batch['gt_heimap'] # loss_stats
                #org_img = inputs[0].transpose(2, 0, 1)
                #cv2.imshow('org_img',org_img)
                #cv2.waitKey(200)
                output, losses = model(inputs, class_masks, center_masks, width_masks, height_masks) #batch
                #loss = loss.mean()
                loss = losses['loss']
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #pbar = tqdm(data_loader, total=len(data_loader), desc="Epoch {} {}".format(epoch, phase), ncols=0)
            # for i, (inputs, class_masks, center_masks, width_masks, height_masks) in enumerate(pbar):
            #########################




                #inputs = inputs.to(device)
                #class_masks = class_masks.to(device)
                #center_masks = center_masks.to(device)
                #width_masks = width_masks.to(device)
                #height_masks = height_masks.to(device)

                #optimizer.zero_grad()

                # compute output
                #with torch.set_grad_enabled(phase == 'train'):
                #    outputs, losses = model(inputs, class_masks, center_masks, width_masks, height_masks)

                #if phase == 'train':
                #    loss = losses['loss']
                #    loss.backward()
                #    optimizer.step()

                for k, v in losses.items():
                    batch_metrics.add(phase, k, v.item())
                if phase =='train':
                    pbar.set_postfix(**{k: "{:.5f}".format(meter.avg) for k, meter in batch_metrics[phase].items()})

            for key, meter in batch_metrics[phase].items():
                epoch_metrics.add(phase, key, meter.avg)
                if phase=='train':
                    print('{}_{}'.format(phase, key), meter.avg)

            #if phase == 'valllllll' and (epoch% 10==0):
                # monitor the val metrics
            #    print ('Iam hereeeeeeee')
            #    best_epoch_index = epoch_metrics['val']['loss'].best()[1]
            #    if best_epoch_index == epoch:
            #        has_waited = 1
            #        best_model_wts = copy.deepcopy(model.state_dict())
            #        print("Saving the best model state dict")
            #    else:
            #        if has_waited >= earlystop_patience:
            #            print("** Early stop in training: {} waits **".format(has_waited))
            #            stop_training = True

            #        has_waited += 1

            #    if type(lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
            #        lr_scheduler.step(epoch_metrics['val']['loss'].value)
            #    else:
            #        lr_scheduler.step()

        #print()  # end of epoch
        #if stop_training:
        #    break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


#import models

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#num_class = 6
#lr_factor = 0.1
#lr_patience = 1
#lr = 1e-4

# model = models.InstanceUNet(num_class).to(device)
# model = models.ObjectDetectUNet(num_class).to(device)
#model = models.WidthHeightUNet(num_class).to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                          #factor=lr_factor,
                                                          #patience=lr_patience,
                                                          #verbose=True)

#model = train_model(model, optimizer, lr_scheduler, dataloaders, device)
#torch.save(model, './models/model_last.pth')