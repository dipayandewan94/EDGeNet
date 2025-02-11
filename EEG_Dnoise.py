
from cProfile import label
import csv
import torch
import PIL
import os
import sys
import time
import imageio
import copy
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
# import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.autograd import Variable, Function

from torch.utils import data as D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tslearn.metrics import SoftDTWLossPyTorch
# from nibabel.testing import data_path
from tqdm import tqdm as tq
# from natsort import natsorted
from itertools import zip_longest
import glob
import pytorch_ssim

from EDGeNet import EDGeNet

import time

from train_utils.eval_func import CC, RRMSE_spectral, RRMSE_temporal
from train_utils.dataloader import EEGDataset
import datetime
import wandb

wandb.login()


device = 'cuda'
os.environ["device"] = device
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

random.seed(53)
np.random.seed(53)
torch.manual_seed(53)
torch.cuda.manual_seed(53)

# final_record = np.load('norm_record.npy')

# final_record = np.load('final_record.npy')
# print('final records: ', len(final_record))

# config = dict(
#     epochs=400,
#     batch_size=256,
#     learning_rate=0.005,
#     #learning_rate_d=0.0001,
#     bridge = True,
#     skip_conn =  True,
#     up_mode = "pixelshuffle",
#     pool = 'conv',
#     residual_enc = False,
#     residual_dec = True,
#     causal_enc = False,
#     causal_dec = True,
#     conv_encoder = 'depthwise',
#     k_sz=5,
#     cmp_ratio = 1,
#     width=[8,16,32,64],
#     lr_reduce = 0.8,
#     wait = 10
#     )

config = dict(
    epochs=400,
    batch_size=1024,
    learning_rate=0.005,
    #learning_rate_d=0.0001,
    bridge = True,
    skip_conn =  True,
    up_mode = "pixelshuffle",
    pool = 'conv',
    residual_enc = True,
    residual_dec = True,
    causal_enc = False,
    causal_dec = False,
    conv_encoder = 'normal',
    k_sz=5,
    cmp_ratio = 1,
    width=[4,8,16,24],
    downsample = [False, False,False],
    lr_reduce = 0.8,
    wait = 20
    )

run = wandb.init(project="EEG_Denoise", config=config)
config = wandb.config

# save_path = '/home/dipayan/QR/'
# save_path_all = os.path.join(save_path, 'model_2_alt_1.h5')
# save_path_dis = os.path.join(save_path, 'model_dis_2_alt_1.h5')
save_path = '/storage/dipayan/EEG_denoise/ELiTNet/'
save_path_all = os.path.join(save_path, 'ELiTNet_1c_k_5_normalized on test data_brige_skip_res_4_8_16_24.pth')
# save_path_dis = os.path.join(save_path, 'ELiTNet_causal_k_7_brige_skip_caus_dis.h5')

# data_path_h = '/home/dipayan/EEG_denoise/'

def min_max_normalize(data, axis=0):
    """
    Apply min-max normalization to the data along the specified axis.
    
    Parameters:
    data (numpy array): The input data.
    axis (int): The axis along which to normalize.
    
    Returns:
    numpy array: The normalized data.
    """
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    return (data - min_val) / (max_val - min_val)

# def min_max_normalize(data, artifact, axis=0):
#     """
#     Apply min-max normalization to the data along the specified axis.
    
#     Parameters:
#     data (numpy array): The input data.
#     axis (int): The axis along which to normalize.
    
#     Returns:
#     numpy array: The normalized data.
#     """
#     min_val = np.min(artifact, axis=axis, keepdims=True)
#     max_val = np.max(artifact, axis=axis, keepdims=True)
#     data = (data - min_val) / (max_val - min_val)
#     artifact = (artifact - min_val) / (max_val - min_val)
#     return data, artifact, min_val, max_val

# def minmax_torch(data, artifact, axis=0):
#     """
#     Apply min-max normalization to the data along the specified axis using PyTorch.
    
#     Parameters:
#     data (torch.Tensor): The input data.
#     artifact (torch.Tensor): The artifact data used for computing min and max.
#     axis (int): The axis along which to normalize.
    
#     Returns:
#     tuple: normalized data, normalized artifact, min values, max values.
#     """
#     min_val = torch.min(artifact, dim=axis, keepdim=True)[0]
#     max_val = torch.max(artifact, dim=axis, keepdim=True)[0]
#     data_norm = (data - min_val) / (max_val - min_val)
#     artifact_norm = (artifact - min_val) / (max_val - min_val)
#     return data_norm, artifact_norm, min_val, max_val

### missing sequence
batch = config.batch_size

EEG_artifact_train = np.load('/storage/dipayan/data_preprocessed_python/EEG_artifact_train.npy')
EEG_artifact_val = np.load('/storage/dipayan/data_preprocessed_python/EEG_artifact_val.npy')
EEG_artifact_test = np.load('/storage/dipayan/data_preprocessed_python/EEG_artifact_test.npy')
EEG_clean_train = np.load('/storage/dipayan/data_preprocessed_python/clean_EEG_train.npy')
EEG_clean_val = np.load('/storage/dipayan/data_preprocessed_python/clean_EEG_val.npy')
EEG_clean_test = np.load('/storage/dipayan/data_preprocessed_python/clean_EEG_test.npy')

# Normalize the datasets
EEG_artifact_train = min_max_normalize(EEG_artifact_train, axis=2)
EEG_artifact_val = min_max_normalize(EEG_artifact_val, axis=2)
EEG_artifact_test = min_max_normalize(EEG_artifact_test, axis=2)
EEG_clean_train = min_max_normalize(EEG_clean_train, axis=2)
EEG_clean_val = min_max_normalize(EEG_clean_val, axis=2)
EEG_clean_test = min_max_normalize(EEG_clean_test, axis=2)



EEG_artifact_train_single = np.expand_dims(np.vstack(EEG_artifact_train), axis=1)
EEG_clean_train_single = np.expand_dims(np.vstack(EEG_clean_train), axis=1)
EEG_artifact_val_single = np.expand_dims(np.vstack(EEG_artifact_val), axis=1)
EEG_clean_val_single = np.expand_dims(np.vstack(EEG_clean_val), axis=1)
EEG_artifact_test_single = np.expand_dims(np.vstack(EEG_artifact_test), axis=1)
EEG_clean_test_single = np.expand_dims(np.vstack(EEG_clean_test), axis=1)

del EEG_artifact_train, EEG_artifact_val, EEG_artifact_test, EEG_clean_train, EEG_clean_val, EEG_clean_test


# train_dataset = EEGDataset(EEG_artifact_train_single, EEG_clean_train_single)
# val_dataset = EEGDataset(EEG_artifact_val_single, EEG_clean_val_single)
# test_dataset = EEGDataset(EEG_artifact_test_single, EEG_clean_test_single)

# # Create DataLoaders
# val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=16)
# test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=16)

# samples = 0.2
# num_train_samples = len(train_dataset)
# indices = list(range(num_train_samples))
# split = int(np.floor(samples * num_train_samples))

train_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_train_single,EEG_clean_train_single), batch_size = batch, shuffle = True, pin_memory=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_val_single,EEG_clean_val_single), batch_size = batch, shuffle = False, pin_memory=True, num_workers=16)
test_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_test_single,EEG_clean_test_single), batch_size = batch, shuffle = False, pin_memory=True, num_workers=16)

# train_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_train,EEG_clean_train), batch_size = batch, shuffle = True, pin_memory=True, num_workers=8)
# val_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_val,EEG_clean_val), batch_size = batch, shuffle = False, pin_memory=True, num_workers=8)
# test_loader = torch.utils.data.DataLoader(EEGDataset(EEG_artifact_test,EEG_clean_test), batch_size = batch, shuffle = False, pin_memory=True, num_workers=8)

net = EDGeNet(in_c=1, n_classes=1, layers=config.width, downsample=config.downsample, k_sz=config.k_sz, up_mode=config.up_mode , pool=config.pool, conv_bridge=config.bridge, shortcut=True, 
                       skip_conn=config.skip_conn, residual_enc= config.residual_enc, causal_enc=config.causal_enc, 
                       residual_dec=config.residual_dec, causal_dec=config.causal_dec,conv_encoder=config.conv_encoder).to(device)

#net.load_state_dict(torch.load(save_path_all))
# optimizer and loss 

optimizer   = optim.Adam(net.parameters(), lr=config.learning_rate) #Adam SGD #, momentum = 0.5
schedule1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_reduce, patience=config.wait, verbose=True)



ssim_cal = pytorch_ssim.SSIM_cal(size_average = True)
soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)
criterion = nn.MSELoss()

iterations= config.epochs   ## number of epochs

Filler_Loss_v = []  

best_test_loss = 1000

for epoch in tq(range(iterations)):
    # np.random.shuffle(indices)
    # train_sampler = SubsetRandomSampler(indices[:split])
    # train_loader = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler, pin_memory=True, num_workers=16)
    
    filler_Loss_t = 0
    ssim_total_t = 0
    total_Loss_t = 0
    ssim_t = 0
    print('number of epoch: ', epoch)
    # for i, data in enumerate(train_record):
    net.train(True)
    tik = datetime.datetime.now()
    print('Tik: ',tik)
    for batch_idx, (train_data, train_target) in tq(enumerate(train_loader)):
        data = copy.deepcopy(train_data)
        target = copy.deepcopy(train_target)
        data = data.to(device)  # data shape = 1,12,1024
        target = target.to(device)

        
        #print(data.shape)
        rec_data = net(data)  # rec data = 1,12,1024
        
        min_val, _ = torch.min(rec_data, axis=2, keepdims=True)
        max_val, _ = torch.max(rec_data, axis=2, keepdims=True)
        norm_rec_data =  (rec_data - min) / (max - min)
        #norm_rec_data = F.sigmoid(rec_data)

        ssim_out_t_measure = ssim_cal(norm_rec_data, target)
        
        optimizer.zero_grad()
        #loss_ae = ssim_loss(norm_rec_data, target)
        #loss_ae = criterion(norm_rec_data, target)
        loss_ae = soft_dtw_loss(norm_rec_data, target).mean()
        total_loss = loss_ae
        #total_loss_value = total_loss.item()
        total_loss.backward()
        optimizer.step()
        
        total_Loss_t+=total_loss.item()
        filler_Loss_t+=loss_ae.item()
        ssim_t+=ssim_out_t_measure.cpu().detach().numpy()
        
    # Calculate average losses and SSIM
    avg_total_loss_t = total_Loss_t / len(train_loader)
    avg_ssim_t = ssim_t / len(train_loader)
    

    wandb.log({
        "epoch": epoch,
        "lr": optimizer.param_groups[0]['lr'],
        "Total_Loss_t": avg_total_loss_t,
        "SSIM_t": avg_ssim_t
    }) 
    learning_rate = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}/{iterations}, Learning Rate: {learning_rate:.6f}, Model Loss: {avg_total_loss_t:.4f},  SSIM: {avg_ssim_t:.4f}')
        ############## Validation ###################
    #filler_Loss_v = 0
    ssim_total_v = 0
    total_loss_v = 0
    ssim_v = 0
    rrmse_t, rrmse_s, cc = 0, 0, 0
    
    print('number of epoch: ', epoch)
    # for i, data_v in enumerate(val_record):
    net.eval()                      # evalution of model
    with torch.no_grad():
        for batch_idx, (val_data, val_target) in tq(enumerate(val_loader)):
            data_v = copy.deepcopy(val_data)
            target_v = copy.deepcopy(val_target)
            # data_v_miss[:,all_leads,:] = 0

            tik = time.time()
            # data= np.expand_dims(data,axis=0)  # when training with out batch
            data_v= data_v.to(device)  # data shape = 1,12,1024
            target_v = target_v.to(device)
            #print(Train_images.shape)
            rec_data_v = net(data_v)  # rec data = 1,12,1024
            
            ## new norm method

            #norm_rec_data_v = F.sigmoid(rec_data_v)
            min_val, _ = torch.min(rec_data_v, axis=2, keepdims=True)
            max_val, _ = torch.max(rec_data_v, axis=2, keepdims=True)
            norm_rec_data_v =  (rec_data_v - min_val) / (max_val - min_val)

            #ssim_out_v_measure,_,_,_,_,_,_,_  = ssim_cal(norm_rec_data_v, data_v_miss) 
            
            #loss_ae_v = ssim_loss(norm_rec_data_v, target_v)
            loss_ae_v = soft_dtw_loss(norm_rec_data_v, target_v).mean()
            total_loss_cuda_v = loss_ae_v
            #total_loss_value_v = total_loss_cuda_v.item()
            total_loss_v+=total_loss_cuda_v.item()
            
            #filler_Loss_v+=loss_ae_v.item()

            value_v  =  ssim_cal(norm_rec_data_v, target_v)
            ssim_v+=value_v.cpu().detach().numpy()
            value_rmse_t = RRMSE_temporal(target_v.cpu().detach(), norm_rec_data_v.cpu().detach())
            value_rmse_s = RRMSE_spectral(target_v.cpu().detach(), norm_rec_data_v.cpu().detach())
            value_cc = CC(target_v.cpu().detach(), norm_rec_data_v.cpu().detach())
            rrmse_t += value_rmse_t
            rrmse_s += value_rmse_s
            cc += value_cc
        total_loss = total_loss_v/len(val_loader)
        schedule1.step(total_loss)
        
        avg_total_loss_v = total_loss_v / len(val_loader)
        avg_ssim_v = ssim_v / len(val_loader)
        avg_rrmse_t = rrmse_t / len(val_loader)
        avg_rrmse_s = rrmse_s / len(val_loader)
        avg_cc = cc / len(val_loader)

        wandb.log({
                "Total_Loss_v": avg_total_loss_v,
                "SSIM_v": avg_ssim_v,
                "CC": avg_cc,
                "RRMSE_temporal": avg_rrmse_t,
                "RRMSE_spectral": avg_rrmse_s
                }) 
        
        print(f'Validation - Epoch {epoch+1}/{iterations}, Model Loss: {avg_total_loss_v:.4f}, SSIM: {avg_ssim_v:.4f}, CC: {avg_cc:.4f}, RRMSE Temporal: {avg_rrmse_t:.4f}, RRMSE Spectral: {avg_rrmse_s:.4f}')
        
        tok = datetime.datetime.now()
        print('Tok: ',tok)
        # print('time epoch: ', tok-tik) 
        if epoch == iterations-1:
            out_v = ssim_cal(norm_rec_data_v, target_v)
            torch.save(net.state_dict(),save_path_all)
        
        if epoch > 0:
            check_v = avg_total_loss_v
            if check_v < best_test_loss :
                print('Model improved from: ', best_test_loss, ' to: ', check_v)
                torch.save(net,save_path_all)
                best_test_loss = check_v

print('recontructed data shape: ', rec_data.shape)
print('data shape: ', data.shape)

