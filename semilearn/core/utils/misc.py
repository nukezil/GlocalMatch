import os
import torch
import torch.nn as nn
import ruamel.yaml as yaml
from torch.utils.tensorboard import SummaryWriter


def over_write_args_from_dict(args, dict):
    """
    overwrite arguments acocrding to config file
    """
    for k in dict:
        setattr(args, k, dict[k])


def over_write_args_from_file(args, yml):
    """
    overwrite arguments according to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def send_model_cuda(args, model):
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            if type(model) is dict:
                for name in model.keys():
                    model[name].cuda(args.gpu)
                    model[name] = nn.SyncBatchNorm.convert_sync_batchnorm(model[name])
                    model[name] = torch.nn.parallel.DistributedDataParallel(model[name], broadcast_buffers=False,
                                                                            find_unused_parameters=True,
                                                                            device_ids=[args.gpu])
            else:
                model.cuda(args.gpu)
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                                  find_unused_parameters=True,
                                                                  device_ids=[args.gpu])
        else:
            # if arg.gpu is None, DDP will divide and allocate batch_size
            # to all available GPUs if device_ids are not set.
            if type(model) is dict:
                for name in model.keys():
                    model[name].cuda()
                    model = torch.nn.parallel.DistributedDataParallel(model[name], broadcast_buffers=False,
                                                                      find_unused_parameters=True)
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                                  find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        if type(model) is dict:
            for name in model.keys():
                model[name] = model[name].cuda(args.gpu)
        else:
            model = model.cuda(args.gpu)
    else:
        if type(model) is dict:
            for name in model.keys():
                model[name] = torch.nn.DataParallel(model[name]).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model


def count_parameters(model):
    # count trainable parameters
    num_param = 0
    if type(model) is dict:
        for name in model.keys():
            num_param += sum(p.numel() for p in model[name].parameters() if p.requires_grad)
    else:
        num_param += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_param


class TBLog:
    """
    Construct tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, log_dict, it, suffix=None, mode="train"):
        """
        Args
            log_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(suffix + key, value, it)


class Bn_Controller:
    """
    Batch Norm controller
    """

    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}


class EMA:
    """
    EMA model
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}