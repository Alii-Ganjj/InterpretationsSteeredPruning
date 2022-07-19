from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0):
    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    return F.sigmoid(y / T)


def hard_concrete(out):
    out_hard = torch.zeros(out.size())
    out_hard[out >= 0.5] = 1
    out_hard[out < 0.5] = 0
    if out.is_cuda:
        out_hard = out_hard.cuda()
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    out_hard = (out_hard - out).detach() + out
    return out_hard


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


class Simplified_Gate(nn.Module):
    def __init__(self, structure=None, T=0.4, base=3, ):
        super(Simplified_Gate, self).__init__()
        self.structure = structure
        self.T = T
        self.base = base

        self.p_list = nn.ModuleList([simple_gate(structure[i]) for i in range(len(structure))])

    def forward(self, ):
        if self.training:
            outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in
                       range(len(self.structure))]
        else:
            outputs = [hard_concrete(gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base)) for i in
                       range(len(self.structure))]

        out = torch.cat(outputs, dim=0)
        return out

    def resource_output(self):
        outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in
                   range(len(self.structure))]

        outputs = [hard_concrete(outputs[i]) for i in
                   range(len(self.structure))]

        out = torch.cat(outputs, dim=0)

        return out

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector


class simple_gate(nn.Module):
    def __init__(self, width):
        super(simple_gate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight


class HyperStructure(nn.Module):
    def __init__(self, structure=None, T=0.4, sparsity=0, base=2, wn_flag=True):
        super(HyperStructure, self).__init__()
        self.bn1 = nn.LayerNorm([256])

        self.T = T
        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)

        self.h0 = torch.zeros(2, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)

        self.inputs.requires_grad = False

        self.sparsity = [sparsity] * len(structure)

        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(256, structure[i], bias=False)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(256, structure[i], bias=False, ) for i
                                in range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.base = base

        self.iteration = 0

    def forward(self):
        self.iteration += 1
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]

        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)

        if not self.training:
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

        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)

        out = hard_concrete(out)
        return out.squeeze()
