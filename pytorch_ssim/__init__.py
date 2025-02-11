import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # create gaussion distribution with sigma defined as 1.5 and add dimension at axis = 1
    # 1d [11, 1]
    # _0D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # mm= matrix multiplication, add 2 dimensions at axis = 0
    # 2d [1, 1, 11, 11]
    # window = Variable(_0D_window.expand(channel, 1, window_size, window_size).contiguous()) # (3,1,11,11) create that window for all the channels
    _1D_window_trans = (torch.transpose(_1D_window, 0, 1)).unsqueeze(0)
    window = Variable(_1D_window_trans.expand(channel, 1, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    # img1 shape = [1, 3, 256, 256], img2 shape = [1, 3, 256, 256]
    mu1 = F.conv1d(img1, window, padding = window_size//2, groups = channel)  # window = weights conv2d input(batch, channel, h,w)
    mu2 = F.conv1d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv1d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv1d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv1d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map1 = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = (1+ssim_map1)/2
    mean_1 = mu1.mean(2).mean(0)
    mean_0 = mu2.mean(2).mean(0)
    sigma_1 = sigma1_sq.mean(2).mean(0)
    sigma_0 = sigma2_sq.mean(2).mean(0)
    
     ## To find max and min
    p, _ = ssim_map.max(2)
    ssim_max,_ = p.max(0)
    s,_ = ssim_map.min(2)
    ssim_min,_ = s.min(0)

    if size_average:
        return ssim_map.mean()
    else:
        return  [ssim_map.mean(2).mean(0),ssim_map.std(2).std(0),ssim_max,ssim_min, mean_1, mean_0, sigma_1, sigma_0]  # ssim_map.mean(1).mean(1).mean(1) [initial]  (changed for have 12 lead SSIM value)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 50, size_average = True):  # changed window_size
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 12 ## define channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIM_cal(torch.nn.Module):
    def __init__(self, window_size = 50, size_average = False):    # changed windo_size
        super(SSIM_cal, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 12 ## define channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 12, size_average = True):
    (_, channel, _) = img1.size()
    window = create_window(window_size, channel) # window of size (3,1,11,11)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)