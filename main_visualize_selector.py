"""
This script visualizes CIFAR-10 samples and their corresponding predicted Amortized Explanations for our model and
REAL-X similar to the Fig.1 of the paper.
"""
import os
import argparse
import random
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt

from Utils.utils import set_logging_settings, add_common_args
from dataset.cifar import CIFAR10_data
from Models.ResNetWithGate import RealTimeSaliencyRBF, RealTimeSaliencyModel

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=11)
parser.add_argument('--verbose', default=1)
parser.add_argument('--num_im_show', default=5)

parser.add_argument('--batch_size', default=1)
parser.add_argument('--val_fraction', default=0.1)

# Model Parameters
parser.add_argument('--selector_type', default='rbf', help='rbf, real-x')
parser.add_argument('--model_type', default='resnet-gate', help='resnet-gate, MobileNetV2')
parser.add_argument('--load_saliency_encoder_from_checkpoint', default=False)
parser.add_argument('--fix_encoder', default=True)
parser.add_argument('--depth', default=56)
parser.add_argument('--initial_sigma', default=None)

parser = add_common_args(parser)
args = parser.parse_args()

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])


# ############################# Loading Data #############################
data = CIFAR10_data(args)
data.prepare_data()
data.setup()
train_dataloader, test_dataloader = data.train_dataloader(), data.test_dataloader()

# ############################# Defining and Loading the Model #############################
if args.selector_type == 'rbf':
    if args.model_type == 'resnet-gate':
        args.pretrained_model_dir = os.path.join('./checkpoints/resnet-56/selector/rbf',
                                                 'class_checkpoint_iter_300.pth')
    elif args.model_type == 'MobileNetV2':
        args.pretrained_model_dir = os.path.join('./checkpoints/mobilenetv2/selector', 'checkpoint_iter_2300.pth')
    model = RealTimeSaliencyRBF(args)
elif args.selector_type == 'real-x':
    args.pretrained_model_dir = os.path.join('./checkpoints/resnet-56/selector/real-x',
                                             'REAL_X_class_checkpoint_iter_300.pth')
    model = RealTimeSaliencyModel(args)
else:
    raise ValueError('selector type must be either rbf or real-x.')

model.load_checkpoint(args.pretrained_model_dir)
model.eval()

# ############################# Visualization #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
n = args.num_im_show
test_images = []
for i, batch in enumerate(test_dataloader):
    if i == n:
        break
    sample_list = []
    x_test, y_test = batch[0], batch[1]
    logging.warning(classes[y_test[0]])
    sample_list.append(x_test[0])
    with torch.no_grad():
        if args.model_type == 'original-resnet-gate':
            mask = model(x_test, y_test)
        else:
            mask, params = model(x_test, y_test)
    binary_mask = torch.clone(mask)
    binary_mask[binary_mask >= 0.5] = 1.
    binary_mask[binary_mask < 0.5] = 0.
    sample_list.append(mask[0])
    sample_list.append(binary_mask[0])
    sample_list.append((x_test * binary_mask)[0])
    test_images.append(sample_list)


fig = plt.figure(figsize=(20, n*5))
for j in range(n):
    for k in range(4):
        axes1 = fig.add_subplot(n, 4, 4*j+k+1)
        if (k != 1) and (k != 2):
            axes1.imshow(test_images[j][k].permute(1, 2, 0) / 5. + 0.5)
        else:
            axes1.imshow(test_images[j][k].squeeze(), cmap='gray')
        axes1.set_axis_off()
fig.subplots_adjust(wspace=0.025, hspace=0.05)
plt.savefig(os.path.join(args.stdout, 'test_images.pdf'), bbox_inches="tight")
plt.show()
plt.close()
