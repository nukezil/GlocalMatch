import os
import numpy as np
from PIL import Image
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data, make_dataset, make_dataset_fnames

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

domain_list = ['synthetic', 'real']

classes_full = ['aeroplane',
                'bicycle',
                'bus',
                'car',
                'horse',
                'knife',
                'motorcycle',
                'person',
                'plant',
                'skateboard',
                'train',
                'truck']

classes_sub = ['aeroplane',
               'bicycle',
               'bus',
               'car',
               'horse',
               'knife']


def make_dataset_with_labels(dir, classnames):
    images = []
    labels = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue

            label = classnames.index(dirname)

            path = os.path.join(root, fname)
            images.append(path)
            labels.append(label)
    return images, labels


def get_visda(args, alg, num_labels, data_dir='./data'):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    num_classes = args.num_classes
    if num_classes == 6:
        classes = classes_sub
    else:
        classes = classes_full

    labeled_domain = args.labeled_domain
    unlabeled_domain = args.unlabeled_domain

    transform_weak = transforms.Compose([
        transforms.Resize([crop_size + 32, crop_size + 32]),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize([crop_size + 32, crop_size + 32]),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize([crop_size + 32, crop_size + 32]),
        transforms.CenterCrop([crop_size, crop_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dirs = {}
    test_dirs = {}

    for domain in domain_list:
        train_dirs[domain] = os.path.join(data_dir, 'visda', domain, 'train')
        test_dirs[domain] = os.path.join(data_dir, 'visda', domain, 'test')

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    train_data_dict = {}
    train_targets_dict = {}
    test_data_dict = {}
    test_targets_dict = {}
    for domain in domain_list:
        train_data_dict[domain], train_targets_dict[domain] = make_dataset_fnames(train_dirs[domain], class_to_idx)
        test_data_dict[domain], test_targets_dict[domain] = make_dataset_fnames(test_dirs[domain], class_to_idx)

    lb_data, lb_targets, _, _ = split_ssl_data(args,
                                               train_data_dict[labeled_domain],
                                               train_targets_dict[labeled_domain],
                                               num_classes=num_classes,
                                               lb_num_labels=num_labels,
                                               domain=labeled_domain)

    in_eval_data, in_eval_targets = test_data_dict[labeled_domain], test_targets_dict[labeled_domain]
    out_eval_data, out_eval_targets = test_data_dict[unlabeled_domain], test_targets_dict[unlabeled_domain]

    ulb_data, ulb_targets = train_data_dict[unlabeled_domain], train_targets_dict[unlabeled_domain]

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    eval_dset = BasicDataset(alg, in_eval_data, in_eval_targets, num_classes, transform_val, False, None, False)
    out_dset = BasicDataset(alg, out_eval_data, out_eval_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset, out_dset

