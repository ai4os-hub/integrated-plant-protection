import os
import threading
from multiprocessing import Pool
import queue
import subprocess
import warnings
import base64
import numpy as np
import requests
from tqdm import tqdm
import cv2

import torch
import random
import joblib
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler

from ai4eosc_uc2.data_utils import prepare_test_data

def _predict(model, device, dataloader):
    res_predictions = []
    res_probs = []
    model.train(False)
    with torch.no_grad():
        for batch in dataloader:
            data = batch.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs).numpy(force=True)
            predictions = ["sick" if prob.item() > 0.5 else "healthy" for prob in probs]
            res_predictions.extend(predictions)
            res_probs.extend(probs.flat)
    return res_predictions, [float(p) for p in res_probs]

def predict(model, device, file_names,  filemode='local', image_size=512, batch_size=16):

    dataloader, _ = prepare_test_data(file_names, filemode, image_size=image_size, batch_size=batch_size)
    result = _predict(model, device, dataloader)
    return result