import os
import argparse

from utils import *
from Models.ResNetWithGate import ResNet
from Models.gate_function import virtual_gate

from Models.hypernet import HyperStructure

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the vgg')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

dir = '/datasets/cifar10/'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

depth = args.depth

if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = ResNet(depth=depth, gate_flag=True)
if args.cuda:
    model.cuda()
model_name = 'resnet'
stat_dict = torch.load('./checkpoint/%s_igp.pth.tar' % (model_name))
model.load_state_dict(stat_dict['net'])
model.cuda()
# resnet56-pruned.pt
width, structure = model.count_structure()

hyper_net = HyperStructure(structure=structure, T=0.4, base=3.0, )
hyper_net.cuda()
hyper_net.load_state_dict(stat_dict['hyper_net'])

vector = stat_dict['arch_vector']
parameters = hyper_net.transfrom_output(vector.detach())

cfg = []
for i in range(len(parameters)):
    cfg.append(int(parameters[i].sum().item()))
newmodel = ResNet(depth=depth, cfg=cfg, gate_flag=True)
newmodel.cuda()

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
start_mask = torch.ones(3)
soft_gate_count = 0
conv_count = 0
end_mask = parameters[soft_gate_count]

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        if layer_id == 2:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            continue
        elif isinstance(old_modules[layer_id + 2], virtual_gate):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            print(m0)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()

        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue

        if isinstance(old_modules[layer_id + 3], virtual_gate):
            print(conv_count)
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            print(m1.weight.data.size())

            m0_next = old_modules[layer_id + 4]
            m1_next = new_modules[layer_id + 4]
            print(m0_next)
            print(m1_next)
            if isinstance(m0_next, nn.Conv2d):
                w1 = m0_next.weight.data[:, idx1.tolist(), :, :].clone()
                m1_next.weight.data = w1.clone()
                print(m1_next.weight.data.size())

            soft_gate_count += 1
            start_mask = end_mask.clone()
            if soft_gate_count < len(parameters):
                end_mask = parameters[soft_gate_count]

            continue
        if isinstance(old_modules[layer_id - 1], virtual_gate):
            continue
        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.

        m1.weight.data = m0.weight.data.clone()

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

model.cpu()
newmodel.cpu()
t_o = print_model_param_nums(model)
t_n = print_model_param_nums(newmodel)
print_model_param_flops(model, input_res=32)
print_model_param_flops(newmodel, input_res=32)

all_parameters = torch.cat(parameters)
print(all_parameters)
pruning_rate = float((all_parameters == 1).sum()) / float(all_parameters.size(0))
print(pruning_rate)

model_new = ResNet(depth=depth, width=2, gate_flag=False, cfg=cfg)

newmodel_ng_ms = list(model_new.modules())
newmodel_ms = list(newmodel.modules())

for m in newmodel_ms:
    if isinstance(m, virtual_gate):
        newmodel_ms.remove(m)

print(len(newmodel_ms))
print(len(newmodel_ng_ms))

for layer_id in range(len(newmodel_ng_ms)):
    m0 = newmodel_ms[layer_id]  # newmodel_ng_ms.remove(m)

    m1 = newmodel_ng_ms[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Conv2d):
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': model_new.state_dict()},
           os.path.join(args.save, './checkpoint/%s_new.pth.tar' % (model_name + str(depth))))
