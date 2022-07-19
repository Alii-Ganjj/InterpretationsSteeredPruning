"""
This script trains ResNet-56 and MobileNetV2 architectures on CIFAR-10.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from Utils.utils import set_logging_settings, add_common_args
from dataset.cifar import CIFAR10_original
from Models.models import BlackBoxModel
from training import train_model_no_val


parser = argparse.ArgumentParser(description='Training the Blackbox model with CIFAR-10')
# Training Parameters
parser.add_argument('--random_seed', default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpu_ids', default=[0, 1, 2, 3])
parser.add_argument('--batch_size', default=128)
parser.add_argument('--num_epochs', default=200)
parser.add_argument('--testing_epoch', default=10)
parser.add_argument('--logging_freq', default=100)
parser.add_argument('--debug', default=False)
parser.add_argument('--metric', default='acc')

# Model Parameters
parser.add_argument('--model_type', default='MobileNetV2', help='resnet, resnet-gate, MobileNetV2')
parser.add_argument('--depth', default=56, help='Is used for the depth of resnet-gate model_type.')

# Optimizer Parameters
parser.add_argument('--lr', default=0.1)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=1e-4)

parser = add_common_args(parser)
args = parser.parse_args()

# ############################# Fixing Seed and Adjusting Logging Settings #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if len(args.gpu_ids) >= 1:
    torch.cuda.manual_seed_all(args.random_seed)

args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])
args.writer = SummaryWriter(args.stdout)

# ############################# Loading Data #############################
if args.debug:
    args.batch_size = 10

data = CIFAR10_original(args)
data.prepare_data()
data.setup()

# ############################# Defining Model #############################
model = BlackBoxModel(args)
model.to(args.device)

# ############################# Training Model #############################
checkpoint_class = train_model_no_val(model, data, args)
checkpoint_class_name = 'checkpoint_iter_{}.pth'.format(checkpoint_class['iter'])
file_name_class = os.path.join(args.checkpoint_dir, checkpoint_class_name)

# ############################# Saving and Evaluating the Checkpoint #############################
logging.warning('Saving Checkpoint: {}'.format(file_name_class))
torch.save(checkpoint_class, file_name_class)
logging.warning('Evaluating the best checkpoint:')
checkpoint_model = checkpoint_class['model']
model.load_state_dict(checkpoint_model)
model.to(args.device)
eval_test = model.eval_model(data.test_dataloader())
for k, v in eval_test.items():
    logging.warning('{}: {:.4f}'.format(k, v))
