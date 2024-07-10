"""
Date: December 2023
Author: Jędrzej Smok 
Email: jsmok@man.poznan.pl
Github: ai4eosc-psnc
"""

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


def prepare_filenames(data_path: str):
    file_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
    return file_paths


def _prepare_test_data(_paths, filemode="local", img_size=512):
    images = [load_image(path, filemode=filemode, img_size=img_size) for path in _paths]
    images = np.asarray(images)
    return images


def _prepare_data(healthy_paths, sick_paths, as_dict=False, img_size=512):
    def equalize_class(healthy_paths, sick_paths):
        X = np.array([*healthy_paths, *sick_paths]).reshape(-1, 1)
        y = [*np.full(len(healthy_paths), 0), *np.full(len(sick_paths), 1)]
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        X_resampled = X_resampled.reshape(-1)

        healthy_paths_resampled = [_x for _x, _class in zip(X_resampled, y_resampled) if _class == 0]
        sick_paths_resampled = [_x for _x, _class in zip(X_resampled, y_resampled) if _class == 1]
        return healthy_paths_resampled, sick_paths_resampled

    def prepare_data_by_class(paths: list, labels_value, img_size, split_part=0.9):
        images = [load_image(path, filemode="local", img_size=img_size) for path in paths]
        images = np.asarray(images)
        np.random.shuffle(images)

        split_index = int(images.shape[0] * split_part)
        images_train = images[:split_index]
        images_test = images[split_index:]

        labels_train = np.full(shape=images_train.shape[0], fill_value=labels_value, dtype=np.uint8)
        labels_test = np.full(shape=images_test.shape[0], fill_value=labels_value, dtype=np.uint8)

        return images_train, images_test, labels_train, labels_test

    def concatenate_data_part(h_images, s_images, h_labels, s_labels):
        images = np.concatenate((h_images, s_images))
        # images = np.stack((images,) * 3, axis=-1)
        labels = np.concatenate((h_labels, s_labels))

        bundle = list(zip(images, labels))
        random.shuffle(bundle)
        bundle = list(zip(*bundle))
        images = np.asarray(bundle[0])
        labels = np.asarray(bundle[1])

        return images, labels

    healthy_paths, sick_paths = equalize_class(healthy_paths, sick_paths)
    h_images_train, h_images_test, h_labels_train, h_labels_test = prepare_data_by_class(
        healthy_paths, labels_value=0, img_size=img_size
    )
    s_images_train, s_images_test, s_labels_train, s_labels_test = prepare_data_by_class(
        sick_paths, labels_value=1, img_size=img_size
    )

    images_train, labels_train = concatenate_data_part(h_images_train, s_images_train, h_labels_train, s_labels_train)
    images_test, labels_test = concatenate_data_part(h_images_test, s_images_test, h_labels_test, s_labels_test)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=images_test.shape[0])
    sss.get_n_splits(images_train, labels_train)
    train_ids, val_ids = next(sss.split(images_train, labels_train))

    images_val, labels_val = images_train[val_ids], labels_train[val_ids]
    images_train, labels_train = images_train[train_ids], labels_train[train_ids]

    if as_dict:
        return {
            "train": {"images": images_train, "labels": labels_train},
            "val": {"images": images_val, "labels": labels_val},
            "test": {"images": images_test, "labels": labels_test},
        }

    return images_train, images_val, images_test, labels_train, labels_val, labels_test


class MyDataset(Dataset):
    def __init__(self, dataset, transform=None, mode="image_label"):
        self.mode = mode
        if self.mode == "image_label":
            self.image_data, self.label_data = dataset
        elif self.mode == "image":
            self.image_data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        image = image.astype(np.uint8)
        if self.transform:
            image = self.transform(image)

        if self.mode == "image_label":
            label = self.label_data[idx]
            return image, label
        elif self.mode == "image":
            return image


class PreprocessImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.image_data, self.paths_img_files = dataset
        self.transform = transform

    def __len__(self):
        return len(self.paths_img_files)

    def __getitem__(self, idx):
        img_path = self.paths_img_files[idx]
        image = self.image_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, img_path


def prepare_preprocess_data(
    file_names: List[str], image_size=512, batch_size=16, num_workers=0
) -> Tuple[DataLoader, int]:
    images = _prepare_test_data(file_names, filemode="local", img_size=image_size)
    test_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )

    dataset = PreprocessImageDataset((images, file_names), test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader, len(dataloader)


def prepare_test_data(
    file_names: List[str], filemode, image_size=512, batch_size=16, num_workers=0
) -> Tuple[DataLoader, int]:
    images = _prepare_test_data(file_names, filemode=filemode, img_size=image_size)
    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = MyDataset(images, test_transform, mode="image")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader, len(dataloader)


def prepare_data(healthy_data_path, sick_data_path, image_size=512, batch_size=16, num_workers=0):
    healthy_paths = prepare_filenames(healthy_data_path)
    sick_paths = prepare_filenames(sick_data_path)
    images_train, images_val, images_test, labels_train, labels_val, labels_test = _prepare_data(
        healthy_paths, sick_paths, img_size=image_size
    )
    data = {
        "train": {"images": images_train, "labels": labels_train},
        "val": {"images": images_val, "labels": labels_val},
        "test": {"images": images_val, "labels": labels_val},
    }
    # print(f"train healthy: {len(data['train']['labels']) - sum(data['train']['labels'])}")
    # print(f"train sick: {sum(data['train']['labels'])}")

    # print(f"test healthy: {len(data['test']['labels']) - sum(data['test']['labels'])}")
    # print(f"test sick: {sum(data['test']['labels'])}")

    # print(f"val healthy: {len(data['val']['labels']) - sum(data['val']['labels'])}")
    # print(f"val sick: {sum(data['val']['labels'])}")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataloaders = {}
    data_len = {}
    for subset in ["train", "val", "test"]:
        if subset == "train" or subset == "val":
            dataset = MyDataset((data[subset]["images"], data[subset]["labels"]), transform, mode="image_label")
        else:
            dataset = MyDataset((data[subset]["images"], data[subset]["labels"]), test_transform, mode="image_label")

        dataloaders[subset] = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)  # TODO
        data_len[subset] = len(dataloaders[subset])

    return dataloaders, data_len


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.

    Example of usage:
        mount_nextcloud('rshare:/data/images', 'my_local_image_path')

    Parameters
    ==========
    * frompath: str, pathlib.Path
        Source folder to be copied
    * topath: str, pathlib.Path
        Destination folder
    """
    command = ["rclone", "copy", f"{frompath}", f"{topath}"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def load_image(filename, filemode="local", img_size=512):
    """
    Function to load a local image path (or an url) into a numpy array.

    Parameters
    ----------
    filename : str
        Path or url to the image
    filemode : {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.

    Returns
    -------
    A numpy array
    """
    if filemode == "local":
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "The local path does not exist or does not correspond to an image: \n {}".format(filename)
            )

    elif filemode == "url":
        try:
            if filename.startswith("data:image"):  # base64 encoded string
                data = base64.b64decode(filename.split(";base64,")[1])
            else:  # normal url

                data = requests.get(filename).content
            data = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception
        except:
            raise ValueError("Incorrect url path: \n {}".format(filename))

    else:
        raise ValueError("Invalid value for filemode.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change from default BGR OpenCV format to Python's RGB format
    image = cv2.resize(image, (int(img_size), int(img_size)))
    return image


def json_friendly(d):
    """
    Return a json friendly dictionary (mainly remove numpy data types)
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)
        elif isinstance(v, (np.ndarray, list)):
            if isinstance(v[0], (np.float32, np.float64)):
                v = np.array(v).astype(float).tolist()
            else:
                v = np.array(v).tolist()
        new_d[k] = v
    return new_d
