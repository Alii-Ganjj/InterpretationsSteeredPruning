from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    return torch.sigmoid(y / T)


def hard_concrete(out):
    out_hard = torch.zeros(out.size())
    out_hard[out >= 0.5] = 1
    if out.is_cuda:
        out_hard = out_hard.cuda()
    out_hard = (out_hard - out).detach() + out
    return out_hard


def truncate_normal_(size, a=-1, b=1):
    values = truncnorm.rvs(a, b, size=size)
    values = torch.from_numpy(values).float()
    return values


class Linear_GW(nn.Module):
    """Linear with custom gradient weights"""
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, sparsity=0.0):
        super(Linear_GW, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sparse_flag = False

        self.sparsity = sparsity
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if sparsity > 0:
            mask = torch.zeros(out_features * in_features)
            perm = torch.randperm(mask.size(0))
            k = int(sparsity * mask.size(0))
            idx = perm[:k]
            r_idx = perm[k:]
            self.weight.requires_grad = False
            weights = self.weight.clone().view(-1)
            weights[r_idx] = 0
            weights = weights.reshape(out_features, in_features)

            self.weight.copy_(weights)
            self.weight.requires_grad = True

            mask[idx] = 1
            mask = mask.reshape(out_features, in_features)
            self.mask = nn.Parameter(mask)
            self.sparse_flag = True
            self.mask.requires_grad = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            nn.init.constant_(self.bias, 0)

    def forward(self, input, gw=1):
        if self.sparse_flag:
            weights = custom_grad_weight.apply(self.weight * self.mask, gw)
        else:
            weights = custom_grad_weight.apply(self.weight, gw)

        return F.linear(input, weights, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None


class HyperStructure(nn.Module):
    def __init__(self, structure=None, T=0.4, sparsity=0, base=2, hard_flag=False, wn_flag=True):
        super(HyperStructure, self).__init__()
        self.bn1 = nn.LayerNorm([256])

        self.T = T
        self.structure = structure
        self.hard_flag = hard_flag
        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)

        self.h0 = torch.zeros(2, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad = False
        print(structure)
        self.sparsity = [sparsity] * len(structure)

        print(self.sparsity)
        if wn_flag:
            linear_list = [weight_norm(Linear_GW(256, structure[i], bias=False, sparsity=self.sparsity[i])) for i in
                           range(len(structure))]
        else:
            linear_list = [Linear_GW(256, structure[i], bias=False, sparsity=self.sparsity[i]) for i in
                           range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(linear_list)
        self.base = base

        self.iteration = 0

        self.gw = torch.ones(len(structure))

    def forward(self):
        self.iteration += 1

        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]

        if self.bn1.weight.is_cuda:
            self.gw = self.gw.cuda()

        outputs = [self.mh_fc[i](outputs[i], self.gw[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)
        if not self.training or self.hard_flag:
            out = hard_concrete(out)

        return out.squeeze()

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

    def resource_output(self):

        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]

        if self.bn1.weight.is_cuda:
            self.gw = self.gw.cuda()

        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)

        out = hard_concrete(out)
        return out.squeeze()

    def collect_gw(self):

        return self.gw.clone()

    def set_gw(self, all_gw):
        self.gw.copy_(all_gw)

    def get_grads(self):
        grads = []

        for i in range(len(self.mh_fc)):
            grads.append((self.mh_fc[i].weight_v.grad.data.abs() * self.mh_fc[i].mask).max())

        print(grads)

    def get_weight_g(self):
        weight_g = self.mh_fc[-1].weight_g.detach().mean()
        print('weight_g: %.4f' % (weight_g))


class PP_Net(nn.Module):
    def __init__(self, structure=None, sparsity=0.4, wn_flag=True):
        super(PP_Net, self).__init__()

        self.bn1 = nn.LayerNorm([128])
        self.bn2 = nn.LayerNorm([128])

        self.structure = structure

        self.Bi_GRU = nn.GRU(128, 64, bidirectional=True)

        self.h0 = torch.zeros(2, 1, 64)
        print(structure)
        self.sparsity = [sparsity] * len(structure)

        print(self.sparsity)
        if wn_flag:
            self.linear_list = [weight_norm(Linear_GW(structure[i], 128, bias=False, sparsity=self.sparsity[i])) for i
                                in range(len(structure))]
        else:
            self.linear_list = [Linear_GW(structure[i], 128, bias=False, sparsity=self.sparsity[i]) for i
                                in range(len(structure))]
        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.pp = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        outputs = self.transfrom_output(x)
        bn = outputs[0].size(0)
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        outputs = [F.relu(self.bn1(outputs[i])) for i in range(len(self.structure))]
        if x.is_cuda:
            self.h0 = self.h0.cuda()

        if len(list(x.size())) == 1:
            outputs = torch.cat(outputs, dim=0).unsqueeze(1)
            h0 = self.h0.clone()
        else:
            outputs = [outputs[i].unsqueeze(0) for i in range(len(outputs))]
            outputs = torch.cat(outputs, dim=0)
            h0 = self.h0.clone().repeat(1, x.size(0), 1)

        outputs, hn = self.Bi_GRU(outputs, h0)
        outputs = outputs.mean(dim=0).squeeze(0)
        outputs = self.pp(self.bn2(outputs))
        return torch.sigmoid(outputs)

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            if len(list(inputs.size())) == 1:
                arch_vector.append(inputs[start:end].unsqueeze(0))
                start = end
            else:
                arch_vector.append(inputs[:, start:end])
                start = end
        return arch_vector


class Episodic_mem(Dataset):
    def __init__(self, K=1000, avg_len=5, structure=None, T=0.4):
        self.K = K
        self.structure = structure
        self.avg_len = avg_len
        self.itr = 0

        len = sum(structure)
        self.k = 0
        self.T = T

        self.sub_arch = torch.zeros(K, len)
        self.acc_list = torch.zeros(K)

        self.local_arch = torch.zeros(len)
        self.local_acc = torch.zeros(1)

    def __getitem__(self, idx):
        current_arch = self.sub_arch[idx]
        current_acc = self.acc_list[idx]

        return current_arch, current_acc

    def insert_data(self, sub_arch, local_acc):

        if self.itr >= self.avg_len:

            self.verify_insert()

            self.itr = 0

            self.local_arch = sub_arch.cpu().data
            self.local_acc = local_acc.cpu().data

            self.itr += 1
        else:
            self.local_arch += sub_arch.cpu().data.clone()
            self.local_acc += local_acc.cpu().data.clone()

            self.itr += 1

    def verify_insert(self):
        if self.k < self.K:
            current_arch = self.local_arch / self.avg_len
            self.sub_arch[self.k, :] = current_arch

            self.acc_list[self.k] = self.local_acc / self.avg_len
            self.k += 1
        elif self.k == self.K:
            acc = self.local_acc / self.avg_len
            current_arch = self.local_arch / self.avg_len

            diff = (acc.unsqueeze(0).expand_as(self.acc_list) - self.acc_list).abs()
            values, index = diff.topk(1, largest=False)

            current_arch = (self.sub_arch[index, :] + current_arch) / 2

            self.sub_arch[index, :] = current_arch

            self.acc_list[index] = (self.acc_list[index] + acc) / 2
            self.k = self.K

    def __len__(self):
        return self.k


if __name__ == '__main__':
    net = HyperStructure()
    y = net()
    print(y)
