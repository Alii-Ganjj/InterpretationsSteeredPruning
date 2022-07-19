from training import *
from utils import *

import torch
import torchvision
import torchvision.transforms as transforms
from dataset.cifar10_dataset import CIFAR10_valdata

import argparse

import torch.optim as optim
from Models.ResNetWithGate import RealTimeSaliencyRBF, RealTimeSaliencyModel
from Models.hypernet import HyperStructure, Simplified_Gate


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--reg_w', default=1, type=float)
parser.add_argument('--base', default=3.0, type=float)
parser.add_argument('--mse_w', default=10.0, type=float)
parser.add_argument('--loss', default='log', type=str)

parser.add_argument('--c_loss', default=False, type=str2bool)
parser.add_argument('--initial_sigma', default=None)
parser.add_argument('--type', default='resnet', type=str)
parser.add_argument('--gau', default=False, type=str2bool)
parser.add_argument('--hs', default=False, type=str2bool)
args = parser.parse_args()
depth = args.depth
model_name = 'resnet'
print('==> Preparing data..')

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

hn_lr = 1e-3
if args.type == 'resnet':
    if args.gau:
        args.pretrained_model_dir = './checkpoint/checkpoint_iter_300.pth'
        model = RealTimeSaliencyRBF(args)
        dir = './val_data/resnet/val_data_gau.pth.tar'
    else:
        args.pretrained_model_dir = './checkpoint/checkpoint_iter_1400.pth'
        model = RealTimeSaliencyModel(args)
        dir = './val_data/val_data.pth.tar'
    args.resnet_dir = './checkpoint/resnet_iter_78200.pth'

    model.load_checkpoint(args.pretrained_model_dir)
    model.resnet.load_checkpoint(args.resnet_dir)

    width, structure = model.resnet.count_structure()

    if args.hs:
        hyper_net = HyperStructure(structure=structure, T=0.4, base=args.base, )
    else:
        hyper_net = Simplified_Gate(structure=structure, T=0.4, base=args.base, )

    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(model.resnet)
    resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                           w=args.reg_w, HN=True, structure=structure, )

elif args.type == 'mbv2':
    if args.gau:
        args.pretrained_model_dir = './checkpoint/mbv2_checkpoint_iter_2300.pth'
        model = RealTimeSaliencyRBF(args)
        dir = './val_data/mbv2/val_data_gau.pth.tar'
    else:
        args.pretrained_model_dir = './checkpoint/mbv2_class_checkpoint_iter_15625.pth'
        model = RealTimeSaliencyModel(args)
        dir = './val_data/mbv2/val_data.pth.tar'

    args.mbv2_dir = './checkpoint/mbv2_iter_70380.pth'

    model.load_checkpoint(args.pretrained_model_dir)
    model.resnet.load_checkpoint(args.mbv2_dir)

    width, structure = model.resnet.count_structure()

    if args.hs:
        hyper_net = HyperStructure(structure=structure, T=0.4, base=args.base, )
    else:
        hyper_net = Simplified_Gate(structure=structure, T=0.4, base=args.base, )

    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(model.resnet)
    resource_reg = Flops_constraint_mobnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                           weight=args.reg_w, HN=True, structure=structure, )

print(dir)

valset = CIFAR10_valdata(dir, test_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/shg/workspace/smooth_prune/datasets/ciar10/', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(hyper_net.parameters(), lr=hn_lr, weight_decay=1e-3)
model.cuda()
hyper_net.cuda()
model.eval()

best_acc = 0
valid(0, model.resnet, testloader, 0, hyper_net=hyper_net, )

for i in range(args.epoch):
    train_IGP_gau(i, model, hyper_net, criterion, valloader, optimizer, resource_reg, args, txt_name=None)
    best_acc = valid(i, model.resnet, testloader, best_acc, hyper_net=hyper_net, model_string='%s_igp' % (args.type),
                     txt_name=None)
