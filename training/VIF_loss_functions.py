import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Resize, Lambda, Normalize
from torch.autograd import Variable
import argparse
from pathlib import Path
from scipy import signal
from enum import Enum


class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1


# References https://github.com/will-leeson/sibyl/blob/main/src/networks/utils/utils.py line 113
# and https://github.com/dmaniry/deepIQA/blob/51dece994d73f43dbd2ea44d3f2ef581014194ad/fr_model.py#L118
class VIFLoss(nn.Module):
    def __init__(self, device):
        super(VIFLoss, self).__init__()
        self.device = device
        self.base_model = torch.load("../weights/model-DAVE2v3-lr1e4-100epoch-batch64-lossMSE-82Ksamples-INDUSTRIALandHIROCHIandUTAH-135x240-noiseflipblur.pt").to(self.device)

    def forward(self, recons, x):
        batch_size = recons.shape[0]
        recons_loss = torch.zeros(batch_size, requires_grad=True).to(device=self.device)
        imgs_recons = Variable(F.split_axis(recons, batch_size, 0), requires_grad=True)
        imgs_orig = Variable(F.split_axis(x, batch_size, 0))
        for i in range(batch_size):
            recons_loss[i] = (1 - self.torch_vif(imgs_recons[i], imgs_orig[i])) / batch_size
        recons_loss = torch.sum(recons_loss)
        pred_recons = self.base_model(recons)
        pred_orig = self.base_model(x)
        prediction_loss = F.mse_loss(pred_orig, pred_recons)
        loss = prediction_loss + recons_loss
        return loss, recons_loss, prediction_loss

    def torch_vif(self, recons, x, sigma_nsq=2):
        channel_vifs = torch.tensor([self.torch_vif_channel(recons[i,:,:], x[i,:,:], sigma_nsq) for i in range(recons.shape[0])], requires_grad=True)
        return torch.mean(channel_vifs)

    def torch_vif_channel(self, recons, x, sigma_nsq):
        EPS = 1e-10
        num = 0.0
        den = 0.0
        for scale in range(1, 5):
            N = 2.0 ** (4 - scale + 1) + 1
            win = fspecial(Filter.GAUSSIAN, ws=N, sigma=N / 5).to(self.device)

            if scale > 1:
                recons = filter2(recons, win, 'valid')
                # recons = recons[2::, 2::]
                x = filter2(x, win, 'valid')
                # x = x[2::, 2::]

            GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(recons, x, win, mode='valid')
            sigmaGT_sq, sigmaP_sq, sigmaGT_P = _get_sigmas(recons, x, win, mode='valid',
                                                           sums=(GT_sum_sq, P_sum_sq, GT_P_sum_mul))

            # sigmaGT_sq[sigmaGT_sq < 0] = 0
            sigmaGT_sq = torch.clamp(sigmaGT_sq, min=0)
            # sigmaP_sq[sigmaP_sq < 0] = 0
            sigmaP_sq = torch.clamp(sigmaP_sq, min=0)

            g = sigmaGT_P / (sigmaGT_sq + EPS)
            sv_sq = sigmaP_sq - g * sigmaGT_P

            g[sigmaGT_sq < EPS] = 0
            sv_sq[sigmaGT_sq < EPS] = sigmaP_sq[sigmaGT_sq < EPS]
            sigmaGT_sq[sigmaGT_sq < EPS] = 0

            g[sigmaP_sq < EPS] = 0
            sv_sq[sigmaP_sq < EPS] = 0

            sv_sq[g < 0] = sigmaP_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= EPS] = EPS

            num += torch.sum(torch.log10(1.0 + (g ** 2.) * sigmaGT_sq / (sv_sq + sigma_nsq)))
            den += torch.sum(torch.log10(1.0 + sigmaGT_sq / sigma_nsq))

        return num / den


####### UTILS FUNCTIONS
conv2d = torch.nn.Conv2d(3, 6, kernel_size=5)
def filter2(img, fltr, mode='same'):
    if len(img.shape) <= 2:
        img = img[None][None]
    if len(fltr.shape) <= 2:
        fltr = fltr[None][None]
    fltr = torch.rot90(fltr, 2)
    return F.conv2d(img, fltr, padding=mode)

def fspecial(fltr, ws, **kwargs):
    if fltr == Filter.UNIFORM:
        return torch.ones((ws,ws))/ ws**2
    elif fltr == Filter.GAUSSIAN:
        ws = int(ws)
        a = torch.tensor([i for i in range(-ws//2 + 1, ws//2 + 1)])
        b = torch.tensor([i for i in range(-ws//2 + 1, ws//2 + 1)])
        x, y = torch.meshgrid(a, b)
        g = torch.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
        g[ g < torch.finfo(g.dtype).eps*g.max() ] = 0
        assert g.shape == (ws,ws)
        den = g.sum()
        if den != 0:
            g /= den
        return g
    return None

def _get_sums(GT,P,win,mode='same'):
    mu1, mu2 = (filter2(GT,win,mode), filter2(P,win,mode))
    return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
    if 'sums' in kwargs:
        GT_sum_sq, P_sum_sq, GT_P_sum_mul = kwargs['sums']
    else:
        GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(GT,P,win,mode)

    return filter2(GT*GT,win,mode) - GT_sum_sq, \
           filter2(P*P,win,mode) - P_sum_sq, \
           filter2(GT*P,win,mode) - GT_P_sum_mul