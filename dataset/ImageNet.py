import logging
import numpy as np

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl


class ImageNet_data(pl.LightningDataModule):
    def __init__(self, args):
        super(ImageNet_data, self).__init__()
        self.args = args
        self.train_set, self.val_set, self.test_set = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(self.args.min_crop_scale, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        self.train_set = torchvision.datasets.ImageFolder(root=self.args.train_dir, transform=self.train_transform)
        self.val_set = torchvision.datasets.ImageFolder(root=self.args.train_dir, transform=self.test_transform)
        self.test_set = torchvision.datasets.ImageFolder(root=self.args.val_dir, transform=self.test_transform)

    def setup(self, stage=None):
        np.random.seed(self.args.seed_data)
        labels = np.unique(np.array(self.train_set.targets))
        num_class = labels.shape[0]
        if self.args.train_fraction == 1.:
            num_val = self.args.val_fraction * len(self.train_set.targets)
            num_val_per_class = num_val / num_class
            idx_val = np.array([])
            for label in labels:
                idx_label = (np.where(np.array(self.train_set.targets) == label))[0]
                idx_sampled = np.random.choice(idx_label, int(num_val_per_class), replace=False)
                idx_val = np.append(idx_val, idx_sampled)
            idx = np.array(range(len(self.train_set)))
            idx_train = np.setxor1d(idx, idx_val).astype(np.int)
            train_sampler = torch.utils.data.SubsetRandomSampler(idx_train)
            val_sampler = torch.utils.data.SubsetRandomSampler(idx_val.astype(np.int))
        else:
            num_train = self.args.train_fraction * len(self.train_set)
            num_val = self.args.val_fraction * len(self.train_set.targets)
            num_train_per_class = num_train / num_class
            num_val_per_class = num_val / num_class
            idx_train, idx_val = np.array([]), np.array([])
            for label in labels:
                idx_label = (np.where(np.array(self.train_set.targets) == label))[0]

                idx_sampled_train = np.random.choice(idx_label, int(num_train_per_class), replace=False)
                idx_train = np.append(idx_train, idx_sampled_train)

                idx_label = np.setxor1d(idx_label, idx_sampled_train)

                idx_sampled_val = np.random.choice(idx_label, int(num_val_per_class), replace=False)
                idx_val = np.append(idx_val, idx_sampled_val)

            train_sampler = torch.utils.data.SubsetRandomSampler(idx_train.astype(np.int))
            val_sampler = torch.utils.data.SubsetRandomSampler(idx_val.astype(np.int))

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.args.batch_size,
                                                        sampler=train_sampler, num_workers=self.args.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.args.batch_size,
                                                      sampler=val_sampler, num_workers=self.args.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.args.batch_size, shuffle=True,
                                                       num_workers=self.args.num_workers)
        logging.warning('Partitioned data. Size: Train: {} \t Validation: {} \t Test: {}'.format(len(idx_train),
                                                                                                 len(idx_val),
                                                                                                 len(self.test_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class ImageNet_original(pl.LightningDataModule):
    def __init__(self, args):
        super(ImageNet_original, self).__init__()
        self.args = args
        self.train_set, self.test_set = None, None
        self.train_loader, self.test_loader = None, None

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        self.train_set = torchvision.datasets.ImageFolder(root=self.args.train_dir, transform=self.train_transform)
        self.test_set = torchvision.datasets.ImageFolder(root=self.args.val_dir, transform=self.test_transform)

    def setup(self, stage=None):
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.args.batch_size,
                                                        num_workers=self.args.num_workers,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=self.args.batch_size,
                                                       num_workers=self.args.num_workers,
                                                       shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader
