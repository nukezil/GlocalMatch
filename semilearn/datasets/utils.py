# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from io import BytesIO

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, num_unlabeled=None,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True, domain=None):
    """
    data & target is splitted into labeled and unlabeled data.
    
    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: number of labeled samples. 
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        num_unlabeled: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """

    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, num_unlabeled, load_exist=load_exist, domain=domain)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)

    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, num_unlabeled=None,
                                  load_exist=True, domain=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    if domain is None:
        domain = 'main'
    dump_dir = os.path.join(base_dir, 'data', 'saved_idx', args.dataset, domain)
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{lb_num_labels}_{num_unlabeled}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{lb_num_labels}_{num_unlabeled}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 

    # get samples per class
    # balanced setting, lb_num_labels is total number of labels for labeled data
    assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
    lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes

    if num_unlabeled is None or num_unlabeled == 'None':
        pass
    else:
        assert num_unlabeled % num_classes == 0, "num_unlabeled must be dividable by num_classes in balanced setting"
        num_unlabeled = num_unlabeled - lb_num_labels
        ulb_samples_per_class = [int(num_unlabeled / num_classes)] * num_classes

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if num_unlabeled is None or num_unlabeled == 'None':
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx


def sample_test_data(args, data, target, num_classes, num_per_class, load_exist=True, domain=None):
    dump_dir = os.path.join(base_dir, 'data', 'saved_idx', args.dataset, domain)
    os.makedirs(dump_dir, exist_ok=True)
    test_dump_path = os.path.join(dump_dir, f'test_{num_per_class}perclass_seed{args.seed}_idx.npy')

    # print(data[:10])

    if os.path.exists(test_dump_path) and load_exist:
        test_idx = np.load(test_dump_path)
        # print(test_idx)
        # print()
        # print(data[test_idx][:10])
        return data[test_idx], target[test_idx]

    test_idx = []

    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        test_idx.extend(idx[:num_per_class])

    if isinstance(test_idx, list):
        test_idx = np.asarray(test_idx)

    np.save(test_dump_path, test_idx)

    return data[test_idx], target[test_idx]

def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fnames(directory, class_to_idx):
    data, targets = [], []
    for class_name in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(directory, class_name)
        for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
            for fname in fnames:
                fpath = os.path.join(root, fname)
                data.append(fpath)
                targets.append(class_idx)

    data = np.array(data, dtype=object)
    targets = np.array(targets)

    return data, targets


def make_dataset(directory, class_to_idx):
    data, targets = [], []
    for class_name in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(directory, class_name)
        for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
            for fname in fnames:
                fpath = os.path.join(root, fname)
                img = np.asarray(pil_loader(fpath))
                data.append(img)
                targets.append(class_idx)
    data = np.array(data, dtype=object)
    targets = np.array(targets)

    return data, targets
