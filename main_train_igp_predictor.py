"""
Code for training our predictor model.
"""
import os
import logging
import argparse
import torch
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from Utils.utils import set_logging_settings, add_common_args
from dataset.cifar import CIFAR10_data
from Models.models import TrainPredictor
from training import train_model


parser = argparse.ArgumentParser()
# Training Parameters
parser.add_argument('--random_seed', default=1)
parser.add_argument('--verbose', default=1)
parser.add_argument('--gpu_ids', default=[])
parser.add_argument('--logging_freq', default=100)
parser.add_argument('--saving_epoch', default=20)
parser.add_argument('--testing_epoch', default=5)
parser.add_argument('--num_epochs', default=300)

parser.add_argument('--batch_size', default=128)
parser.add_argument('--debug', default=False)
parser.add_argument('--val_fraction', default=0.1)
parser.add_argument('--metric', default='loss')

# Model Parameters
parser.add_argument('--selector_type', default='rbf', help='real-x, rbf')
parser.add_argument('--rbf_sig_start', default=5.)
parser.add_argument('--rbf_sig_end', default=11.)
parser.add_argument('--model_type', default='resnet-gate', help='resnet-gate, MobileNetV2')
parser.add_argument('--depth', default=56, help='Is used for the depth of resnet-gate model_type.')
parser.add_argument('--hard_sample', default=False)
parser.add_argument('--temperature', default=0.1)

# Optimizer Parameters
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--betas', default=(0.9, 0.999))
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--momentum', default=0.9)

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
data = CIFAR10_data(args)
data.prepare_data()
data.setup()
np.random.seed(args.random_seed)

# ############################# Defining Model #############################
if args.model_type == 'resnet-gate':
    args.pretrained_model_dir = os.path.join('./checkpoints/resnet-56/blackbox', 'checkpoint_iter_78200.pth')
elif args.model_type == 'MobileNetV2':
    args.pretrained_model_dir = os.path.join('./checkpoints/mobilenetv2/blackbox', 'checkpoint_iter_70380.pth')

model = TrainPredictor(args)

# ############################# Training Model #############################
checkpoint_class = train_model(model, data, args)
checkpoint_class_name = 'class_checkpoint_iter_{}.pth'.format(checkpoint_class['iter'])
file_name_class = os.path.join(args.checkpoint_dir, checkpoint_class_name)

# ############################# Saving and Evaluating the Checkpoint #############################
logging.warning('Saving Checkpoint: {}'.format(file_name_class))
torch.save(checkpoint_class, file_name_class)
logging.warning('Evaluating the last checkpoint:')
checkpoint_model = checkpoint_class['model']
model.load_state_dict(checkpoint_model)
model.to(args.device)
eval_test = model.eval_model(data.test_dataloader())
for k, v in eval_test.items():
    logging.warning('{}: {:.4f}'.format(k, v))
