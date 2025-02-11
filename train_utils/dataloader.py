import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
"""
Author: wolider wong
Date: 2024-1-14
Description: EEG, EOG, EMG data preprocessing to form the dataset for training
cite: EEGDnet: Fusing non-local and local self-similarity for EEG signal denoising with transformer
"""

# Author: woldier wong
# The code here not only include data importing, but also data standardization and the generation of analog noise signals

def RMS(x: np.ndarray):
    """
    Root Mean Squared (RMS)

    :param x: input
    :return:
    """
    x2 = x ** 2  # x^2
    sum_s2 = np.sum(x2, axis=-1, keepdims=True)  # sum
    return (sum_s2 / x.shape[-1]) ** 0.5


def compute_noise_signal(x: np.ndarray, n: np.ndarray, snr: np.ndarray):
    """
    λ is a hyperparameter to control the signal-to-noise ratio (SNR) in the contaminated EEG signal y
    SNR = 10 log( RMS(x) / RMS(λ · n) )

    SNR = 10 log ( RMS(x) / ( λ · RMS(n) )  )

    (SNR / 10 ) ** 10 = RMS(x) / ( λ · RMS(n) )

    y = x + λ · n
    :param x: noise-free signal
    :param n: noise signal
    :param snr:
    :return:
    """
    lamda = RMS(x) / ((10 ** (snr / 10)) * RMS(n))
    return x + lamda * n


def normalize(x: np.ndarray, y: np.ndarray, mean_norm=False):
    """
    In order to facilitate the learning procedure, we normalized the input contaminated EEG segment and the ground-truth
    EEG segment by dividing the standard deviation of contaminated EEG segment according to
    x_bar = x / std(y)
    y_bar = y / std(y)
    :param x: noise-free signal
    :param y: contaminated signal
    :param mean_norm: bool , default false  . If true, will norm mean to 0
    :return:
    """
    mean = y.mean() if mean_norm else 0
    std = y.std(axis=-1, keepdims=True)
    x_bar = (x - mean) / std
    y_bar = (y - mean) / std
    # min_x = np.min(x, axis=1, keepdims=True)
    # max_x = np.max(x, axis=1, keepdims=True)
    # min_y = np.min(y, axis=1, keepdims=True)
    # max_y = np.max(y, axis=1, keepdims=True)
    # x_bar = (x - min_x) / (max_x - min_x)
    # y_bar = (y - min_y) / (max_y - min_y)
    return x_bar, y_bar, std


def data_prepare(EEG_all: np.ndarray, noise_all: np.ndarray, combin_num: int, train_num: int, test_num: int):
    # The code here not only include data importing,
    # but also data standardization and the generation of analog noise signals

    # First of all, if we just divide the data into training set and test set according to train_num,test_num,
    # then the coverage of the samples in the training set and test set may not be comprehensive,
    # because we should do a disruptive operation before dividing the data.

    # a random seed to 109(this number can be chosen at random, the realization of the choice of 109 just to have a good feeling about the number),
    # to ensure that each time the random result is the same
    np.random.seed(109)
    # disruptive element
    # disorder the elements of an array
    np.random.shuffle(EEG_all)
    np.random.shuffle(noise_all)

    # Get x, and n for the training and test sets
    eeg_train, eeg_test = EEG_all[0:train_num, :], EEG_all[train_num:train_num + test_num, :]
    noise_train, noise_test = noise_all[0:train_num, :], noise_all[train_num:train_num + test_num, :]

    # Repeat the dataset combin_num times to accumulate noise of different intensities.
    # shape [train_num * combin_num, L] , [test_num * combin_num, L]
    eeg_train, eeg_test = np.repeat(eeg_train, combin_num, axis=0), np.repeat(eeg_test, combin_num, axis=0)
    noise_train, noise_test = np.repeat(noise_train, combin_num, axis=0), np.repeat(noise_test, combin_num, axis=0)

    #################################  simulate noise signal of training set  ##############################

    # create random number between -7dB ~ 2dB
    snr_table = np.linspace(-7, -7 + combin_num, combin_num)  # a shape of [combin_num]
    snr_table = snr_table.reshape((1, -1))  # reshape to [1, combin_num]
    num_table = np.zeros((train_num, 1))  # reshape to [train_num, 1]
    snr_table = snr_table + num_table  # broadcast to [train_num, combin_num]
    snr_table = snr_table.reshape((-1, 1))  # match samples [train_num * combin_num, 1]
    eeg_train_y = compute_noise_signal(eeg_train, noise_train, snr_table)

    # normalize
    EEG_train_end_standard, noiseEEG_train_end_standard, train_std = normalize(eeg_train, eeg_train_y)

    #################################  simulate noise signal of test  ##############################
    snr_table = np.linspace(-7, -7 + combin_num, combin_num)  # a shape of [combin_num]
    snr_table = snr_table.reshape((1, -1))  # reshape to [1, combin_num]
    num_table = np.zeros((test_num, 1))  # reshape to [test_num, 1]
    snr_table = snr_table + num_table  # broadcast to [test_num, combin_num]
    snr_table = snr_table.reshape((-1, 1))  # match samples [test_num * combin_num, 1]
    eeg_test_y = compute_noise_signal(eeg_test, noise_test, snr_table)
    EEG_test_end_standard, noiseEEG_test_end_standard, test_std = normalize(eeg_test, eeg_test_y)

    return noiseEEG_train_end_standard, EEG_train_end_standard, noiseEEG_test_end_standard, EEG_test_end_standard, train_std, test_std


class EEGDataset(Dataset):
    def __init__(self, EEG_artifact, clean_EEG):
        self.EEG_artifact = torch.tensor(EEG_artifact, dtype=torch.float32)
        self.clean_EEG = torch.tensor(clean_EEG, dtype=torch.float32)


    def __len__(self):
        return len(self.EEG_artifact)

    def __getitem__(self, idx):
        artifact = self.EEG_artifact[idx]
        clean = self.clean_EEG[idx]
        
        return artifact, clean
    
# Define a custom PyTorch dataset class
class HuggingFaceEEGDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_tensor = torch.tensor(item['x'], dtype=torch.float32).unsqueeze(0)  # Adjust the key based on your dataset
        target_tensor = torch.tensor(item['y'], dtype=torch.float32).unsqueeze(0)  # Adjust the key based on your dataset
        
        # Apply min-max normalization
        # input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
        # target_tensor = (target_tensor - target_tensor.min()) / (target_tensor.max() - target_tensor.min())

        return input_tensor, target_tensor
    
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        
        input1 = item1['input']
        target1 = item1['target']
        input2 = item2['input']
        target2 = item2['target']
        
        return input1, target1, input2, target2
    
def custom_collate_fn(batch):
    inputs1, targets1, inputs2, targets2 = zip(*batch)
    combined_inputs = torch.cat(inputs1 + inputs2, dim=0).unsqueeze(1)
    combined_targets = torch.cat(targets1 + targets2, dim=0).unsqueeze(1)
    return combined_inputs, combined_targets