from utils import TrainVal_split
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset.cifar10_dataset import ORI_CIFAR10
from Models.ResNetWithGate import RealTimeSaliencyModel

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=1)
parser.add_argument('--user', default='shangqian')
parser.add_argument('--verbose', default=1)
parser.add_argument('--num_im_show', default=20)

parser.add_argument('--num_classes', default=10)
# Model Parameters
parser.add_argument('--type', default='resnet', type=str)
parser.add_argument('--load_saliency_encoder_from_checkpoint', default=False)
parser.add_argument('--fix_encoder', default=True)
parser.add_argument('--depth', default=56)

args = parser.parse_args()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

])

trainset = ORI_CIFAR10(root='/home/shg/workspace/smooth_prune/datasets/ciar10/', train=True, download=True,
                       transform=test_transform)
train_sampler, val_sampler = TrainVal_split(trainset, 0.05, shuffle_dataset=True)
val_loader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=4, sampler=val_sampler)

tqdm_loader = tqdm(val_loader)

if args.type == 'resenet':
    args.pretrained_model_dir = './checkpoint/checkpoint_iter_1400.pth'
    args.resnet_dir = './checkpoint/resnet_iter_78200.pth'
    model = RealTimeSaliencyModel(args)
    model.load_checkpoint(args.pretrained_model_dir)
    model.resnet.load_checkpoint(args.resnet_dir)
elif args.type == 'mbv2':
    args.pretrained_model_dir = './checkpoint/mbv2_class_checkpoint_iter_15625.pth'
    args.mbv2_dir = './checkpoint/mbv2_iter_70380.pth'
    model = RealTimeSaliencyModel(args)
    model.load_checkpoint(args.pretrained_model_dir)
    model.resnet.load_checkpoint(args.mbv2_dir)

model.cuda()
model.eval()

all_masks = []
all_images = []
all_labels = []
for batch_idx, (inputs, ori_inputs, targets) in enumerate(tqdm_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    with torch.no_grad():
        mask = model(inputs, targets)
    all_masks.append(mask.cpu())
    all_images.append(ori_inputs.cpu())
    all_labels.append(targets.cpu())
all_masks = torch.cat(all_masks, dim=0)
all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)

mask_dataset = {}
mask_dataset['masks'] = all_masks
mask_dataset['images'] = all_images
mask_dataset['labels'] = all_labels

import os

directory = './val_data/%s/' % (args.type)
if not os.path.exists(directory):
    os.makedirs(directory)

torch.save(mask_dataset, directory + 'val_data.pth.tar')
