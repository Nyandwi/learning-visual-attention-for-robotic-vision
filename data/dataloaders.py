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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", DEVICE)
import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel


# Create directories for the train and validation images and annotations
DATA_DIR = '<path to the dataset>'
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR,"images", 'train')
TRAIN_ANNOT_DIR = os.path.join(DATA_DIR, "annotations", "train")
VALID_IMAGE_DIR = os.path.join(DATA_DIR, "images", "test")
VALID_ANNOT_DIR = os.path.join(DATA_DIR, "annotations", 'val')
TEST_IMAGE_DIR = os.path.join(DATA_DIR,"images", 'val')


# Create directories for the train and validation images and annotations
DATA_DIR = '<path to the MIT dataset>'
MIT1003_TEST_IMAGE_DIR = os.path.join(DATA_DIR,"images")
MIT1003_TEST_ANNOT_DIR = os.path.join(DATA_DIR, "annotations")

# Transforms for the train images and annotations
train_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

annot_transforms = transforms.Compose([transforms.ToTensor()])

# Transforms for the train images and annotations
mit1003_test_transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize(size=(224, 224)),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

mit1003_annot_transforms = transforms.Compose([    transforms.ToTensor(),transforms.Resize(size=(224, 224))])

# Salicon Dataset class
class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None, annot_transforms =None, train=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.annot_transforms = annot_transforms
        self.train = train

        # This line of code returns a sorted list of full paths to each image in the directory
        self.img_paths = list(map(lambda fname: os.path.join(self.image_dir, fname), sorted(os.listdir(self.image_dir))))

        # Annotated images
        self.annotation_paths = list(map(lambda fname: os.path.join(self.annotation_dir, fname), sorted(os.listdir(self.annotation_dir))))

        # Load the center bias template calculated using the MIT1003 dataset
        # centerbias_template = np.load('/content/drive/MyDrive/centerbias_mit1003.npy')
        centerbias_template = np.load('path to center bias npy file')
        # 480, 640

        # The size of the center bias changes depending on the size of the image. For each image we have to change this value torch_image.shape[1]
        self.centerbias = zoom(centerbias_template, (480/centerbias_template.shape[0], 640/centerbias_template.shape[1]), order=0, mode='nearest')
        # renormalize log density
        self.centerbias -= logsumexp(self.centerbias)
        # image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        # centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.train:
            # image = self.img_paths[idx]
            if self.annot_transforms(Image.open(self.img_paths[idx])).shape[0] ==1:
                # self.img_paths[idx] = torch.stack((image, image, image), dim=0).squeeze()
                image = self.annot_transforms(Image.open(self.img_paths[idx]))
                return torch.stack((image, image, image),dim=0).squeeze(), self.annot_transforms(Image.open(self.annotation_paths[idx])), self.annot_transforms(self.centerbias)
        return self.transforms(Image.open(self.img_paths[idx])), self.annot_transforms(Image.open(self.annotation_paths[idx])), self.annot_transforms(self.centerbias)
    
# MIT1003 dataset class
class MIT1003Dataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None, annot_transforms =None, train=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.annot_transforms = annot_transforms
        self.train = train

        # This line of code returns a sorted list of full paths to each image in the directory
        self.img_paths = list(map(lambda fname: os.path.join(self.image_dir, fname), sorted(os.listdir(self.image_dir))))

        # Annotated images
        self.annotation_paths = list(map(lambda fname: os.path.join(self.annotation_dir, fname) if not "Pts" in fname else "null", sorted(os.listdir(self.annotation_dir))))
        self.annotation_paths = [p for p in self.annotation_paths if p != "null"]

        # Load the center bias template calculated using the MIT1003 dataset
        # centerbias_template = np.load('/content/drive/MyDrive/centerbias_mit1003.npy')
        centerbias_template = np.load('<path to the centerbias file>')
        # 480, 640

        # The size of the center bias changes depending on the size of the image. For each image we have to change this value torch_image.shape[1]
        self.centerbias = zoom(centerbias_template, (480/centerbias_template.shape[0], 640/centerbias_template.shape[1]), order=0, mode='nearest')
        # renormalize log density
        self.centerbias -= logsumexp(self.centerbias)
        # image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
        # centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

        assert len(self.annotation_paths) == len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx])), self.annot_transforms(Image.open(self.annotation_paths[idx])), self.annot_transforms(self.centerbias)

    def collate_fn(batch):
        # print(batch.shape)
        return torch.stack([data[0] for data in batch]), torch.stack([data[1] for data in batch]), torch.stack([data[2] for data in batch])
    
# Dataset and dataloader class for the transalnet model
def preprocess_img(img_dir, channels=3):
    """Function that converts an input image to the size expected by the model"""
    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    shape_r = 288
    shape_c = 384
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img

    return img_padded

def postprocess_img(pred, org_dir):
    '''Method that upsamples the map produced by the model back to the same size as the input'''
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img

# Dataset class for the transalnet model
class MyDataset(Dataset):
    """Load dataset."""

    def __init__(self,stimuli_dir, saliency_dir,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.ids = ids
        self.image_dir = stimuli_dir
        self.annotation_dir = saliency_dir
        # self.fixation_dir = fixation_dir
        self.transform = transform

        # This line of code returns a sorted list of full paths to each image in the directory
        self.img_paths = list(map(lambda fname: os.path.join(self.image_dir, fname), sorted(os.listdir(self.image_dir))))

        # Annotated images
        self.annotation_paths = list(map(lambda fname: os.path.join(self.annotation_dir, fname), sorted(os.listdir(self.annotation_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        image   = preprocess_img(self.img_paths[idx])

        # image   = Image.open(self.img_paths[idx]).convert('RGB')

        img = np.array(image) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        if self.transform:
           img = self.transform(image)



        # smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        # saliency = Image.open(self.annotation_paths[idx])
        saliency   = preprocess_img(self.annotation_paths[idx], channels=1)


        smap = np.expand_dims(np.array(saliency) / 255., axis=0)
        smap = torch.from_numpy(smap)

        sample = {'image': img, 'saliency': smap}

        return sample

# SALICON Dataloader 
train_dataset = MyDataset(TRAIN_IMAGE_DIR,TRAIN_ANNOT_DIR,train_transforms, annot_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, drop_last =False, num_workers =2)

valid_dataset = MyDataset(VALID_IMAGE_DIR, VALID_ANNOT_DIR, train_transforms, annot_transforms)
valid_loader = DataLoader(valid_dataset, batch_size=32, drop_last =False, num_workers=2)

# MIT1003 dataset and DataLoader 
mit1003_test_dataset = MIT1003Dataset(MIT1003_TEST_IMAGE_DIR,MIT1003_TEST_ANNOT_DIR, mit1003_test_transforms, mit1003_annot_transforms)
mit1003_test_loader = DataLoader(mit1003_test_dataset, batch_size=32, drop_last =False, num_workers =2)

train_set = Dataset(stimuli_dir= '<location of the images>', 
                      saliency_dir= '<file path to the saliency maps>',
                      transform=transforms.Compose([
                      # transforms.Resize((384, 288)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ]))

val_set = Dataset(stimuli_dir= '<file path to the images>', 
                      saliency_dir= '<file path to the saliency maps>',
                      transform=transforms.Compose([
                      # transforms.Resize((384, 288)),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]))

# Train and Validation dataloaders
dataloaders = {'train':DataLoader(train_set, batch_size=4,shuffle=True, num_workers=4),
               'val':DataLoader(val_set, batch_size=32,shuffle=False, num_workers=4)}

