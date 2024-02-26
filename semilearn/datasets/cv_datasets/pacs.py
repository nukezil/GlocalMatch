import os
import numpy as np

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data, sample_test_data, find_classes, make_dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

domain_list = ['art', 'cartoon', 'photo', 'sketch']


def get_pacs(args, alg, num_labels, data_dir='./data'):
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

    train_dirs = {}
    test_dirs = {}
    for domain in domain_list:
        train_dirs[domain] = os.path.join(data_dir, 'pacs', domain, 'train')
        test_dirs[domain] = os.path.join(data_dir, 'pacs', domain, 'test')

    classes, class_to_idx = find_classes(train_dirs['art'])

    train_data_dict = {}
    train_targets_dict = {}
    test_data_dict = {}
    test_targets_dict = {}
    for domain in domain_list:
        train_data_dict[domain], train_targets_dict[domain] = make_dataset(train_dirs[domain], class_to_idx)
        test_data_dict[domain], test_targets_dict[domain] = make_dataset(test_dirs[domain], class_to_idx)

    lb_data_dict = {}
    lb_targets_dict = {}
    ulb_data_dict = {}
    ulb_targets_dict = {}
    for domain in domain_list:
        lb_data_dict[domain], lb_targets_dict[domain], \
            ulb_data_dict[domain], ulb_targets_dict[domain] = split_ssl_data(args,
                                                                             train_data_dict[domain],
                                                                             train_targets_dict[domain],
                                                                             num_classes=7,
                                                                             lb_num_labels=num_labels,
                                                                             domain=domain)

    lb_data, lb_targets = lb_data_dict[labeled_domain], lb_targets_dict[labeled_domain]
    in_eval_data, in_eval_targets = test_data_dict[labeled_domain], test_targets_dict[labeled_domain]

    ulb_data = np.concatenate([ulb_data_dict[domain] for domain in domain_list])
    ulb_targets = np.concatenate([ulb_targets_dict[domain] for domain in domain_list])

    out_eval_data = np.concatenate([test_data_dict[domain] for domain in domain_list if domain != labeled_domain])
    out_eval_targets = np.concatenate([test_targets_dict[domain] for domain in domain_list if domain != labeled_domain])

    all_eval_data = np.concatenate([in_eval_data, out_eval_data])
    all_eval_targets = np.concatenate([in_eval_targets, out_eval_targets])

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    eval_dset = BasicDataset(alg, in_eval_data, in_eval_targets, num_classes, transform_val, False, None, False)
    out_dset = BasicDataset(alg, out_eval_data, out_eval_targets, num_classes, transform_val, False, None, False)
    all_eval_dset = BasicDataset(alg, all_eval_data, all_eval_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset, out_dset, all_eval_dset
