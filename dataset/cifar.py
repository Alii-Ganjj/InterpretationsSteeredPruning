import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np


class CIFAR10_data(pl.LightningDataModule):
    def __init__(self, args):
        super(CIFAR10_data, self).__init__()
        self.args = args
        self.cifar_train, self.cifar_val, self.cifar_test = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        self.train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        self.cifar_train = torchvision.datasets.CIFAR10(root=self.args.data_dir, train=True, download=True,
                                                        transform=self.train_transform)
        self.cifar_val = torchvision.datasets.CIFAR10(root=self.args.data_dir, train=True, download=True,
                                                      transform=self.test_transform)
        self.cifar_test = torchvision.datasets.CIFAR10(root=self.args.data_dir, train=False, download=True,
                                                       transform=self.test_transform)

    def setup(self, stage=None):
        np.random.seed(self.args.seed_data)
        labels = np.unique(np.array(self.cifar_train.targets))
        num_class = labels.shape[0]
        num_val = self.args.val_fraction * len(self.cifar_train)
        num_val_per_class = num_val / num_class
        idx_val = np.array([])
        for label in labels:
            idx_label = (np.where(np.array(self.cifar_train.targets) == label))[0]
            idx_sampled = np.random.choice(idx_label, int(num_val_per_class), replace=False)
            idx_val = np.append(idx_val, idx_sampled)
        idx = np.array(range(len(self.cifar_train)))
        idx_train = np.setxor1d(idx, idx_val).astype(np.int)
        train_sampler = torch.utils.data.SubsetRandomSampler(idx_train)
        val_sampler = torch.utils.data.SubsetRandomSampler(idx_val.astype(np.int))
        self.train_loader = torch.utils.data.DataLoader(self.cifar_train, batch_size=self.args.batch_size,
                                                        sampler=train_sampler, num_workers=self.args.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.cifar_val, batch_size=self.args.batch_size,
                                                      sampler=val_sampler, num_workers=self.args.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.cifar_test, batch_size=self.args.batch_size, shuffle=False,
                                                       num_workers=self.args.num_workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class CIFAR10_original(pl.LightningDataModule):
    def __init__(self, args):
        super(CIFAR10_original, self).__init__()
        self.args = args
        self.cifar_train, self.cifar_test = None, None
        self.train_loader, self.test_loader = None, None

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def prepare_data(self):
        self.cifar_train = torchvision.datasets.CIFAR10(root=self.args.data_dir, train=True, download=True,
                                                        transform=self.train_transform)
        self.cifar_test = torchvision.datasets.CIFAR10(root=self.args.data_dir, train=False, download=True,
                                                       transform=self.test_transform)

    def setup(self, stage=None):
        self.train_loader = torch.utils.data.DataLoader(self.cifar_train,
                                                        batch_size=self.args.batch_size,
                                                        num_workers=self.args.num_workers,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.cifar_test,
                                                       batch_size=self.args.batch_size,
                                                       num_workers=self.args.num_workers,
                                                       shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader
