import os
import time
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision
from scipy.ndimage import zoom
from scipy.special import logsumexp
from collections import OrderedDict
import torchvision
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F
import functools
import math
from pysaliency.roc import general_roc
from pysaliency.numba_utils import auc_for_one_positive

def _general_auc(positives, negatives):
    if len(positives) == 1:
        return auc_for_one_positive(positives[0], negatives)
    else:
        return general_roc(positives, negatives)[0]


def log_likelihood(log_density, fixation_mask, weights=None):
    if weights is None:
       weights = torch.ones(log_density.shape[0])

    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()

    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask
    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)
    
    ll = torch.mean(
        weights * torch.sum(log_density * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return (ll + np.log(log_density.shape[-1] * log_density.shape[-2])) / np.log(2)


def nss(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    mean, std = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    nss = torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True) / fixation_count
    )
    return nss


def auc(log_density, fixation_mask, weights=None):
    weights = len(weights) * weights / weights.sum()

    # TODO: This doesn't account for multiple fixations in the same location!
    def image_auc(log_density, fixation_mask):
        if isinstance(fixation_mask, torch.sparse.IntTensor):
            dense_mask = fixation_mask.to_dense()
        else:
            dense_mask = fixation_mask

        positives = torch.masked_select(log_density, dense_mask.type(torch.bool)).detach().cpu().numpy().astype(np.float64)
        negatives = log_density.flatten().detach().cpu().numpy().astype(np.float64)

        auc = _general_auc(positives, negatives)

        return torch.tensor(auc)

    return torch.mean(weights.cpu() * torch.tensor([
        image_auc(log_density[i], fixation_mask[i]) for i in range(log_density.shape[0])
    ]))
    
    
def train_epoch(model, dataset, optimizer, device):
    model.train()
    losses = []
    batch_weights = []
    pbar = tqdm(dataset)
    for batch in pbar:
        optimizer.zero_grad()
        x, y, z = batch
        image = x.to(device)
        fixation = y.to(device)
        gauss = z.mean(dim=0).to(device)
        weights = torch.ones(x.shape[0]).to(device)
        
        log_density = model(image, gauss)
        # print("Train loop",log_density.shape)

        loss = -log_likelihood(log_density, fixation, weights=weights)
        losses.append(loss.detach().cpu().numpy())

        batch_weights.append(weights.detach().cpu().numpy().sum())

        pbar.set_description('{:.05f}'.format(np.average(losses, weights=batch_weights)))

        loss.backward()

        optimizer.step()

    return np.average(losses, weights=batch_weights)


def eval_epoch(model, dataset, device, metrics= None):
    model.eval()
    if metrics is None:
        metrics = ['LL', 'NSS', 'AUC'] # 'IG',

    metric_scores = {}
    metric_functions = {
    'LL': log_likelihood,
    'NSS': nss,
    'AUC': auc,
    }
    batch_weights = []

    with torch.no_grad():
        pbar = tqdm(dataset)
        for batch in pbar:
            optimizer.zero_grad()
            x, y, z = batch
            image = x.to(device)
            fixation = y.to(device)
            centerbias = z.mean(dim=0).to(device)
            weights = torch.ones(x.shape[0]).to(device)
            
            log_density = model(image, centerbias)

            for metric_name, metric_fn in metric_functions.items():
                if metric_name not in metrics:
                    continue
                metric_scores.setdefault(metric_name, []).append(metric_fn(log_density, fixation, weights=weights).detach().cpu().numpy())
            batch_weights.append(weights.detach().cpu().numpy().sum())

            for display_metric in ['LL', 'NSS', 'AUC']:
                if display_metric in metrics:
                    pbar.set_description('{} {:.05f}'.format(display_metric, np.average(metric_scores[display_metric], weights=batch_weights)))
                    break

    data = {metric_name: np.average(scores, weights=batch_weights) for metric_name, scores in metric_scores.items()}

    return data



class SaliencyLoss(nn.Module):
    '''Defines loss functions for the Transalnet model'''
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, loss_type='cc'):
        losses = []
        if loss_type == 'cc':
            for i in range(labels.shape[0]): # labels.shape[0] is batch size
                loss = loss_CC(preds[i],labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv':
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i],labels[i])
                losses.append(loss)

        elif loss_type == 'sim':
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i],labels[i])
                losses.append(loss)

        elif loss_type == 'nss':
            for i in range(labels.shape[0]):
                loss = loss_NSS(preds[i],labels[i])
                losses.append(loss)
            
        return torch.stack(losses).mean(dim=0, keepdim=True)
        
        
def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map/torch.sum(pred_map)
    gt_map = gt_map/torch.sum(gt_map)
    div = torch.sum(torch.mul(gt_map, torch.log(eps + torch.div(gt_map,pred_map+eps))))
    return div 
        
    
def loss_CC(pred_map,gt_map):
    gt_map_ = (gt_map - torch.mean(gt_map))
    pred_map_ = (pred_map - torch.mean(pred_map))
    cc = torch.sum(torch.mul(gt_map_,pred_map_))/torch.sqrt(torch.sum(torch.mul(gt_map_,gt_map_))*torch.sum(torch.mul(pred_map_,pred_map_)))
    return cc


def loss_similarity(pred_map,gt_map):
    gt_map = (gt_map - torch.min(gt_map))/(torch.max(gt_map)-torch.min(gt_map))
    gt_map = gt_map/torch.sum(gt_map)
    
    pred_map = (pred_map - torch.min(pred_map))/(torch.max(pred_map)-torch.min(pred_map))
    pred_map = pred_map/torch.sum(pred_map)
    
    diff = torch.min(gt_map,pred_map)
    score = torch.sum(diff)
    
    return score
    
    
def loss_NSS(pred_map,fix_map):
    '''ground truth here is fixation map'''

    pred_map_ = (pred_map - torch.mean(pred_map))/torch.std(pred_map)
    mask = fix_map.gt(0)
    score = torch.mean(torch.masked_select(pred_map_, mask))
    return score

# Utilities for the transalnet models