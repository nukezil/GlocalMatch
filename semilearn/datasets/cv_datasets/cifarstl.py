import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data, sample_test_data, find_classes, make_dataset, make_dataset_fnames

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_cifarstl(args, alg, num_labels, data_dir='./data'):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    num_classes = 9
    labeled_domain = args.labeled_domain
    single_base = args.single_base
    all_out = args.all_out
    num_unlabeled = args.num_unlabeled

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar_train_dir = os.path.join(data_dir, 'cifarstl/cifar/train')
    cifar_test_dir = os.path.join(data_dir, 'cifarstl/cifar/test')

    classes, class_to_idx = find_classes(cifar_train_dir)
    cifar_train_data, cifar_train_targets = make_dataset_fnames(cifar_train_dir, class_to_idx)
    cifar_test_data, cifar_test_targets = make_dataset_fnames(cifar_test_dir, class_to_idx)

    stl_train_dir = os.path.join(data_dir, 'cifarstl/stl/train')
    stl_test_dir = os.path.join(data_dir, 'cifarstl/stl/test')
    stl_train_data, stl_train_targets = make_dataset_fnames(stl_train_dir, class_to_idx)
    stl_test_data, stl_test_targets = make_dataset_fnames(stl_test_dir, class_to_idx)

    cifar_lb_data, cifar_lb_targets, cifar_ulb_data, cifar_ulb_targets = split_ssl_data(args,
                                                                                        cifar_train_data,
                                                                                        cifar_train_targets,
                                                                                        num_classes=9,
                                                                                        lb_num_labels=num_labels,
                                                                                        num_unlabeled=num_unlabeled[0],
                                                                                        domain='cifar')

    stl_lb_data, stl_lb_targets, stl_ulb_data, stl_ulb_targets = split_ssl_data(args,
                                                                                stl_train_data,
                                                                                stl_train_targets,
                                                                                num_classes=9,
                                                                                lb_num_labels=num_labels,
                                                                                num_unlabeled=num_unlabeled[1],
                                                                                domain='stl')

    cifar_eval_data, cifar_eval_targets = sample_test_data(args, cifar_test_data, cifar_test_targets,
                                                           num_classes=9, num_per_class=500, domain='cifar')

    stl_eval_data, stl_eval_targets = sample_test_data(args, stl_test_data, stl_test_targets,
                                                       num_classes=9, num_per_class=500, domain='stl')

    if labeled_domain == 'cifar':
        lb_data, lb_targets = cifar_lb_data, cifar_lb_targets
        source_eval_data, source_eval_targets = cifar_eval_data, cifar_eval_targets
    else:
        lb_data, lb_targets = stl_lb_data, stl_lb_targets
        source_eval_data, source_eval_targets = stl_eval_data, stl_eval_targets

    if single_base:
        if labeled_domain == 'cifar':
            ulb_data, ulb_targets = cifar_ulb_data, cifar_ulb_targets
        else:
            ulb_data, ulb_targets = stl_ulb_data, stl_ulb_targets
    elif all_out:
        if labeled_domain == 'cifar':
            ulb_data, ulb_targets = stl_ulb_data, stl_ulb_targets
        else:
            ulb_data, ulb_targets = cifar_ulb_data, cifar_ulb_targets
    else:
        ulb_data = np.concatenate((cifar_ulb_data, stl_ulb_data))
        ulb_targets = np.concatenate((cifar_ulb_targets, stl_ulb_targets))

    all_eval_data = np.concatenate((cifar_eval_data, stl_eval_data))
    all_eval_targets = np.concatenate((cifar_eval_targets, stl_eval_targets))

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    eval_dset = BasicDataset(alg, source_eval_data, source_eval_targets, num_classes, transform_val, False, None, False)
    all_eval_dset = BasicDataset(alg, all_eval_data, all_eval_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset, all_eval_dset
