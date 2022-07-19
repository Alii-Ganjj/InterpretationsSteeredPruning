import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset.ImageNetLabels import ImageNet_labels
from collections import OrderedDict
from torch.distributions import Bernoulli
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from Models.sampler import RandomBernoulliSampler
from imagenet_models.resnet_gate import my_resnet_dict
from Models.MobileNet import MobileNetV2
from imagenet_models.gate_function import virtual_gate
from Models.saliency_architecture import encoder_resnet_dict, UpSampleBlock, encoder_mobilenet_dict
from torchvision.models.resnet import resnet34, resnet50, resnet101

import numpy as np

original_resnet_dict = {'resnet-34': resnet34, 'resnet-50': resnet50, 'resnet-101': resnet101}
resnet_channels_dict = {'resnet-34': [512, 256, 128, 64], 'resnet-50': [2048, 1024, 512, 256],
                        'resnet-101': [2048, 1024, 512, 256]}
mobilenet_channels_dict = {'MobileNetV2': [1280, 96, 32, 24]}


class BlackBoxModel(nn.Module):
    def __init__(self, args):
        super(BlackBoxModel, self).__init__()
        self.args = args
        self.model = None
        self.build_network()

    def build_network(self):
        if 'resnet' in self.args.model_type:
            self.model = my_resnet_dict[self.args.model_type]()
            pretrained_model = original_resnet_dict[self.args.model_type](pretrained=True)
            model_modules = list(self.model.modules())
            model_non_gate_modules = [x for x in model_modules if not isinstance(x, virtual_gate)]
            pretrained_model_modules = list(pretrained_model.modules())
            for layer_id in range(len(model_non_gate_modules)):
                m0 = pretrained_model_modules[layer_id]
                m1 = model_non_gate_modules[layer_id]
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

        elif self.args.model_type == 'MobileNetV2':
            self.model = MobileNetV2(gate_flag=False)
            self.load_mobilenet_imagenet_checkpoint()
        else:
            raise ValueError("Blackbox's model architecture should be either ResNet or MobileNet")

        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

    def eval_model(self, test_dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                # if i == 5:
                #     break
                x, y = data[0].to(self.args.device), data[1].to(self.args.device)
                y_pred = self.model(x)
                loss = self.class_loss(y_pred, y)
                total_loss += loss * y.shape[0]
                _, label_pred = torch.max(y_pred, dim=1)
                total += y.shape[0]
                correct += (label_pred == y).sum().item()
        total_loss = total_loss / total
        accuracy = correct / total
        return {'loss': total_loss, 'acc': accuracy}

    def add_eval_results_to_writer(self, eval_results, partition='Val', iteration=None, epoch=None):
        if (iteration is None) and (epoch is None):
            raise ValueError('One of the iteration or epoch values should be not None.')
        if iteration is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Iter'.format(partition, k), v, iteration)

        elif epoch is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Epoch'.format(partition, k), v, epoch)

    def forward(self, x):
        return self.model(x)

    def load_mobilenet_imagenet_checkpoint(self):
        pretrained_checkpoint = torch.load('./checkpoints/mobilenetv2/blackbox/mobilenetv2_1.0-0c6065bc.pth',
                                           map_location='cpu')
        orig_mobilenet = MobileNetV2(gate_flag=False)
        orig_mobilenet.load_state_dict(pretrained_checkpoint)
        model_modules = list(self.model.modules())
        model_non_gate_modules = [x for x in model_modules if not isinstance(x, virtual_gate)]
        pretrained_model_modules = list(orig_mobilenet.modules())
        for layer_id in range(len(model_non_gate_modules)):
            m0 = pretrained_model_modules[layer_id]
            m1 = model_non_gate_modules[layer_id]
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


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.model = None
        self.build_network()

    def build_network(self):
        if 'resnet' in self.args.model_type:
            if self.args.start_predictor_from_pretrained:
                logging.warning('Starting predictor from a model pretrained on ImageNet.')
                self.model = original_resnet_dict[self.args.model_type](pretrained=True)
            else:
                logging.warning('Starting predictor from a random initialized model.')
                self.model = original_resnet_dict[self.args.model_type]()

        elif self.args.model_type == 'MobileNetV2':
            self.model = MobileNetV2(gate_flag=True)
            if self.args.start_predictor_from_pretrained:
                logging.warning('Starting predictor from a model pretrained on ImageNet.')
                self.load_mobilenet_imagenet_checkpoint()
            else:
                logging.warning('Starting predictor from a random initialized model.')
        else:
            raise ValueError("Blackbox's model architecture should be either ResNet or MobileNet")

        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('######## Loading Pretrained Predictor ########')
        logging.warning('loading the pretrained Predictor checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):  # Data Parallel mode on GPUs.
            for k, v in checkpoint['model'].items():
                if not ('predictor' in k):
                    continue
                name = k.replace('predictor.model.', '')
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
        else:
            for k, v in checkpoint['model'].items():
                if not ('predictor' in k):
                    continue
                if 'predictor.model.module.' in k:
                    name = k.replace('predictor.model.module.', '')
                else:
                    name = k.replace('predictor.model.', '')
                new_dict[name] = v
            self.model.load_state_dict(new_dict)

    def load_mobilenet_imagenet_checkpoint(self):
        pretrained_checkpoint = torch.load('./checkpoints/mobilenetv2/blackbox/mobilenetv2_1.0-0c6065bc.pth',
                                           map_location='cpu')
        orig_mobilenet = MobileNetV2(gate_flag=False)
        orig_mobilenet.load_state_dict(pretrained_checkpoint)
        model_modules = list(self.model.modules())
        model_non_gate_modules = [x for x in model_modules if not isinstance(x, virtual_gate)]
        pretrained_model_modules = list(orig_mobilenet.modules())
        for layer_id in range(len(model_non_gate_modules)):
            m0 = pretrained_model_modules[layer_id]
            m1 = model_non_gate_modules[layer_id]
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


class TrainPredictor(nn.Module):
    def __init__(self, args):
        super(TrainPredictor, self).__init__()
        self.args = args
        self.predictor = Predictor(self.args)
        self.predictor.to(self.args.device)

        self.bb_model = BlackBoxModel(self.args)
        self.bb_model.to(self.args.device)

        self.loss_fn = F.kl_div
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=self.args.lr, betas=self.args.betas,
                                    weight_decay=self.args.weight_decay)
        if self.args.hard_sample and (self.args.selector_type == 'real-x'):
            self.mask_generator = RandomBernoulliSampler(self.args)
        self.scheduler = None

    def training_step(self, x, y, iteration):
        self.predictor.train()
        self.bb_model.eval()
        self.optimizer.zero_grad()

        if self.args.selector_type == 'real-x':
            if self.args.hard_sample:
                mask = self.mask_generator(x)
            else:
                n, c, h, w = x.shape
                sampler = RelaxedBernoulli(temperature=self.args.temperature, probs=torch.ones((n, 1, h, w)) * 0.5)
                mask = sampler.sample().to(self.args.device)
        elif self.args.selector_type == 'rbf':
            n, c, h, w = x.shape
            probs, params = self.calculate_rbf(n, h, w)
            if self.args.hard_sample:
                sampler = Bernoulli(probs=probs)
                mask = sampler.sample().to(self.args.device)
            else:
                sampler = RelaxedBernoulli(temperature=self.args.temperature, probs=probs)
                mask = sampler.sample().to(self.args.device)
        else:
            raise NotImplementedError

        with torch.no_grad():
            y_model = self.bb_model(x)

        # for i in range(x.shape[0]):
        #     self.visualize_sample(mask[i], x[i], y[i], params=params[i])

        x = x * mask
        y_pred = self.predictor(x)

        loss = self.loss_fn(F.log_softmax(y_pred, 1),
                            F.log_softmax(y_model, 1),
                            reduction='batchmean',
                            log_target=True)
        loss.backward()
        self.optimizer.step()
        loss_dict = {'loss': loss}
        self.add_train_losses_to_writer(loss_dict, iteration)
        return loss_dict

    def eval_model(self, test_dataloader):
        self.predictor.eval()
        self.bb_model.eval()
        total_loss = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                x = data[0].to(self.args.device)
                if self.args.selector_type == 'real-x':
                    if self.args.hard_sample:
                        mask = self.mask_generator(x)
                    else:
                        n, c, h, w = x.shape
                        sampler = RelaxedBernoulli(temperature=self.args.temperature,
                                                   probs=torch.ones((n, 1, h, w)) * 0.5)
                        mask = sampler.sample().to(self.args.device)
                elif self.args.selector_type == 'rbf':
                    n, c, h, w = x.shape
                    probs, params = self.calculate_rbf(n, h, w)
                    if self.args.hard_sample:
                        sampler = Bernoulli(probs=probs)
                        mask = sampler.sample().to(self.args.device)
                    else:
                        sampler = RelaxedBernoulli(temperature=self.args.temperature, probs=probs)
                        mask = sampler.sample().to(self.args.device)
                y_model = self.bb_model(x)
                x = x * mask
                y_pred = self.predictor(x)
                loss = self.loss_fn(F.log_softmax(y_pred, 1),
                                    F.log_softmax(y_model, 1),
                                    reduction='batchmean',
                                    log_target=True)
                total_loss += loss * x.shape[0]
                total += x.shape[0]
        total_loss = total_loss / total
        return {'loss': total_loss}

    def calculate_rbf(self, n, h, w):
        xy = (torch.rand((n, 2)) * 2. - 1.) * 108. + 112.
        sigma = torch.rand(n) * (self.args.rbf_sig_end - self.args.rbf_sig_start) + \
                self.args.rbf_sig_start
        maps = []
        params = torch.cat((xy, torch.unsqueeze(sigma, 1)), 1)
        for i in range(n):
            x_c, y_c = self.coordinate_arrays(h, w)
            map = (1 / (2 * torch.tensor(np.pi) * (sigma[i] ** 2))) * \
                  torch.exp((-1. / (2 * (sigma[i] ** 2))) * (((x_c - xy[i, 0]) ** 2) + ((y_c - xy[i, 1]) ** 2)))
            maps.append(map.unsqueeze(0) / map.detach().max())

        out_maps = torch.stack(maps)
        return out_maps, params

    @staticmethod
    def visualize_sample(s, x, y, partition='test', params=None):
        s, x, y = s.detach(), x.detach(), y.detach()
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x.permute((1, 2, 0)) / 5 + 0.5)
        axarr[0].axis('off')
        axarr[1].imshow(s.squeeze(), cmap='gray')
        axarr[1].axis('off')
        axarr[2].imshow((x * s).permute((1, 2, 0)) / 5 + 0.5)
        axarr[2].axis('off')
        if params is not None:
            f.suptitle("{}: {}, {}, x: {:.1f}, y: {:.1f}, sig: {:.1f}".
                       format(partition, y.item(), ImageNet_labels[y.item()], params[0], params[1], params[2]))
        else:
            f.suptitle("{}: {}, {}".format(partition, y.item(), ImageNet_labels[y.item()]))
        plt.show()

    @staticmethod
    def coordinate_arrays(h, w):
        y_coordinates = (torch.arange(float(w))).repeat((w, 1))
        x_coordinates = torch.transpose((torch.arange(float(h))).repeat((h, 1)), 1, 0)
        return x_coordinates, y_coordinates

    def add_train_losses_to_writer(self, loss_dict, iteration):
        for k, v in loss_dict.items():
            self.args.writer.add_scalar('Train/{}_Iter'.format(k), v, iteration)

    def add_eval_results_to_writer(self, eval_results, partition='Val', iteration=None, epoch=None):
        if (iteration is None) and (epoch is None):
            raise ValueError('One of the iteration or epoch values should be not None.')
        if iteration is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Iter'.format(partition, k), v, iteration)

        elif epoch is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Epoch'.format(partition, k), v, epoch)


class RealTimeSaliencyRBF(nn.Module):
    def __init__(self, args):
        super(RealTimeSaliencyRBF, self).__init__()
        self.args = args
        self.fix_encoder = self.args.fix_encoder
        self.upsample_dims = None
        self.build_network()

        if self.args.initial_sigma:
            # Initializing bias of the 'sigma' parameter to proper number to accelerate training.
            self.saliency_chans.bias.data[2].fill_(self.args.initial_sigma)

    def build_network(self):
        if 'resnet' in self.args.model_type:
            self.upsample_dims = resnet_channels_dict[self.args.model_type]
            self.encoder = encoder_resnet_dict[self.args.model_type]()
            if self.args.load_saliency_encoder_from_pretrained:
                pretrained_model = original_resnet_dict[self.args.model_type](pretrained=True)
                model_modules = list(self.encoder.modules())
                model_non_gate_modules = [x for x in model_modules if not isinstance(x, virtual_gate)]
                pretrained_model_modules = list(pretrained_model.modules())
                for layer_id in range(len(model_non_gate_modules)):
                    m0 = pretrained_model_modules[layer_id]
                    m1 = model_non_gate_modules[layer_id]
                    if isinstance(m0, nn.BatchNorm2d):
                        m1.weight.data = m0.weight.data.clone()
                        m1.bias.data = m0.bias.data.clone()
                        m1.running_mean = m0.running_mean.clone()
                        m1.running_var = m0.running_var.clone()
                    elif isinstance(m0, nn.Conv2d):
                        m1.weight.data = m0.weight.data.clone()

            upsample_dims = resnet_channels_dict[self.args.model_type]

        elif 'MobileNet' in self.args.model_type:
            self.upsample_dims = mobilenet_channels_dict[self.args.model_type]
            self.encoder = encoder_mobilenet_dict[self.args.model_type](gate_flag=True)
            if self.args.load_saliency_encoder_from_pretrained:
                pretrained_checkpoint = torch.load('./checkpoints/mobilenetv2/blackbox/mobilenetv2_1.0-0c6065bc.pth',
                                                   map_location='cpu')
                orig_mobilenet = MobileNetV2(gate_flag=False)
                orig_mobilenet.load_state_dict(pretrained_checkpoint)
                model_modules = list(self.encoder.modules())
                model_non_gate_modules = [x for x in model_modules if not isinstance(x, virtual_gate)]
                pretrained_model_modules = list(orig_mobilenet.modules())
                for layer_id in range(len(model_non_gate_modules)):
                    m0 = pretrained_model_modules[layer_id]
                    m1 = model_non_gate_modules[layer_id]
                    if isinstance(m0, nn.BatchNorm2d):
                        m1.weight.data = m0.weight.data.clone()
                        m1.bias.data = m0.bias.data.clone()
                        m1.running_mean = m0.running_mean.clone()
                        m1.running_var = m0.running_var.clone()
                    elif isinstance(m0, nn.Conv2d):
                        m1.weight.data = m0.weight.data.clone()

            upsample_dims = mobilenet_channels_dict[self.args.model_type]

        else:
            raise ValueError

        self.uplayer4 = UpSampleBlock(in_channels=upsample_dims[0], out_channels=upsample_dims[1],
                                      passthrough_channels=upsample_dims[1])
        self.uplayer3 = UpSampleBlock(in_channels=upsample_dims[1], out_channels=upsample_dims[2],
                                      passthrough_channels=upsample_dims[2])
        self.uplayer2 = UpSampleBlock(in_channels=upsample_dims[2], out_channels=upsample_dims[3],
                                      passthrough_channels=upsample_dims[3])
        self.embedding = nn.Embedding(self.args.num_classes, upsample_dims[0])
        self.saliency_chans = nn.Conv2d(upsample_dims[3], 3, kernel_size=56, bias=True)

        if self.fix_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, labels):
        if self.fix_encoder:
            self.encoder.eval()
            with torch.no_grad():
                scale1, scale2, scale3, scale4 = self.encoder(x)
        else:
            scale1, scale2, scale3, scale4 = self.encoder(x)

        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4 * em.view(-1, self.upsample_dims[0], 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4 * th

        upsample3 = self.uplayer4(scale4, scale3)
        upsample2 = self.uplayer3(upsample3, scale2)
        upsample1 = self.uplayer2(upsample2, scale1)

        saliency_params = self.saliency_chans(upsample1)
        masks = self.calculate_rbf(saliency_params)
        return masks, saliency_params.squeeze()

    def calculate_rbf(self, saliency_params):
        params = saliency_params.squeeze()
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
        xy = (108. * torch.tanh(params[:, :2] / 108.) + 112.).to(
            self.args.device)  # we use tanh relative to the center of image and add 16 to make
        # coordinates relative to top-left. (the final values will be in the center 28*28 frame of the image)
        sigma = (torch.logaddexp(torch.zeros_like(params[:, 2]), params[:, 2]) + 1e-8).to(
            self.args.device)  # sigma = log(1 + exp(m_x))
        maps = []
        for i in range(params.shape[0]):
            x_c, y_c = self.coordinate_arrays()
            map = (1 / (2 * torch.tensor(np.pi) * (sigma[i] ** 2))) * \
                  torch.exp((-1. / (2 * (sigma[i] ** 2))) * (((x_c - xy[i, 0]) ** 2) + (
                          (y_c - xy[i, 1]) ** 2)))  # Calculating Gaussian density at each pixel.
            new_map = map.unsqueeze(0) / (map.detach().max() + 1e-8)  # Converting gaussian density to RBF.
            if (torch.isnan(new_map)).sum() != 0:
                import pdb
                pdb.set_trace()
            maps.append(new_map)

        out_maps = (torch.stack(maps)).to(self.args.device)
        return out_maps

    def coordinate_arrays(self):
        y_coordinates = ((torch.arange(224.)).repeat((224, 1))).to(self.args.device)
        x_coordinates = (torch.transpose((torch.arange(224.)).repeat((224, 1)), 1, 0)).to(self.args.device)
        return x_coordinates, y_coordinates

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('#' * 20)
        logging.warning('loading pretrained selector from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if 'selector' in k:
                name = k.replace('selector.', '')
                if 'module' in name:
                    name = name.replace('module.', '')
                new_dict[name] = v

        self.load_state_dict(new_dict)


class TrainSelector(nn.Module):

    def __init__(self, args):
        super(TrainSelector, self).__init__()
        self.args = args
        self.lamda = self.args.lamda
        if self.args.selector_type == 'rbf':
            self.selector = RealTimeSaliencyRBF(self.args)
        elif self.args.selector_type == 'real-x':
            raise NotImplementedError
        else:
            raise ValueError

        self.selector.to(self.args.device)

        self.bb_model = BlackBoxModel(self.args)
        self.bb_model.to(self.args.device)

        self.predictor = Predictor(self.args)
        self.predictor.load_checkpoint(self.args.pretrained_predictor_dir)
        self.predictor.to(self.args.device)
        for p in self.predictor.parameters():
            p.requires_grad = False

        self.loss_fn = F.kl_div
        self.optimizer = optim.Adam(self.selector.parameters(), lr=self.args.lr, betas=self.args.betas,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = None

    def training_step(self, x, y, iteration):
        self.bb_model.eval()
        self.predictor.eval()
        self.selector.train()
        self.optimizer.zero_grad()

        with torch.no_grad():
            y_pred_bb_model = self.bb_model(x)

        sel_prob, params = self.selector(x, y)
        sampler = RelaxedBernoulli(temperature=self.args.temperature, probs=sel_prob)
        s_relaxed = sampler.rsample()
        if self.args.hard_sample:
            s = (s_relaxed > 0.5).detach().float().to(self.args.device) - s_relaxed.detach() + s_relaxed
        else:
            s = s_relaxed

        # if iteration % 1 == 0:
        #     n = 0
        #     self.visualize_sample(s[n], x[n], y[n], params[n])

        # Calculate
        # 1. f(s)
        # f_s = self.predictor([x_batch, s], training=True)
        f_s = self.predictor(x * s)

        # Compute the probabilities
        # 1. f(s)
        p_f_s = self.loss_fn(F.log_softmax(f_s, 1),
                             F.log_softmax(y_pred_bb_model, 1),
                             reduction='none',
                             log_target=True).sum(1)

        s_flat = s.flatten(1)
        # Compute the Sparisity Regularization
        # 1. R(s)
        R_s = torch.mean(s_flat, 1)
        # R_s_approx = torch.mean(s_relaxed.flatten(1), 1)
        R_s_approx = torch.mean((s_flat + 0.0005) ** self.args.area_loss_power, 1)

        s_loss = p_f_s + self.lamda * R_s_approx
        s_loss = s_loss.mean()

        if self.args.smoothness_loss_coeff != 0.:
            smooth_loss = self.smoothness_loss(s_relaxed)
            s_loss = s_loss + self.args.smoothness_loss_coeff * smooth_loss  # Maybe we can also try smoothness loss on s.
            if iteration % 50 == 0:
                logging.warning('iter: {}, kl: {:.4f}, Rs: {:.4f}, smooth: {:.4f}'.format(iteration,
                                                                                          p_f_s.mean().item(),
                                                                                          R_s_approx.mean().item(),
                                                                                          smooth_loss.item()))
                log_dict = {'kl': p_f_s.mean().item(), 'Rs': R_s_approx.mean().item(), 'smooth': smooth_loss.item()}
                for k, v in log_dict.items():
                    self.args.writer.add_scalar('Train/{}_Iter'.format(k), v, iteration)

        else:
            if iteration % 50 == 0:
                logging.warning('iter: {}, kl: {:.4f}, Rs: {:.4f}'.format(iteration,
                                                                          p_f_s.mean().item(),
                                                                          R_s_approx.mean().item()))
                log_dict = {'kl': p_f_s.mean().item(), 'Rs': R_s_approx.mean().item()}
                for k, v in log_dict.items():
                    self.args.writer.add_scalar('Train/{}_Iter'.format(k), v, iteration)

        s_loss.backward()
        self.optimizer.step()

        # Calculate Objective Loss
        objective = (p_f_s + self.lamda * R_s).mean()

        loss_dict = {'loss': s_loss, 'objective': objective.item(), 'kl_loss': p_f_s.mean().item()}
        self.add_train_losses_to_writer(loss_dict, iteration)
        return loss_dict

    def eval_model(self, test_dataloader):
        self.predictor.eval()
        self.bb_model.eval()
        self.selector.eval()
        total_loss = 0
        total_kl_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                x, y = data[0].to(self.args.device), data[1].to(self.args.device)
                sel_prob, params = self.selector(x, y)
                # s = (sel_prob > 0.5).detach().float().to(self.args.device)
                s = sel_prob
                s_flat = s.flatten(1)
                y_pred_bb_model = self.bb_model(x)
                _, y_pred_discrete_bb = torch.max(y_pred_bb_model, 1)

                f_s = self.predictor(x * s)
                _, y_pred_selector = torch.max(f_s, 1)

                kl_loss = self.loss_fn(F.log_softmax(f_s, 1),
                                       F.log_softmax(y_pred_bb_model, 1),
                                       reduction='none',
                                       log_target=True).sum(1)

                correct += (y_pred_discrete_bb == y_pred_selector).sum().item()

                R_s = torch.mean(s_flat, 1)
                loss = kl_loss + self.lamda * R_s
                loss = loss.mean()

                total_loss += loss * x.shape[0]
                total_kl_loss += kl_loss.mean() * x.shape[0]
                total += x.shape[0]

        total_loss = total_loss / total
        total_kl_loss = total_kl_loss / total
        return {'loss': total_loss, 'kl_loss': total_kl_loss, 'acc': correct / total}

    @staticmethod
    def visualize_sample(s, x, y, params):
        classes = ImageNet_labels
        s, x, y, params = s.detach(), x.detach(), y.detach(), params.detach()
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x.permute((1, 2, 0)) / 5 + 0.5)
        axarr[0].axis('off')
        axarr[1].imshow(s[0], cmap='gray')
        axarr[1].axis('off')
        axarr[2].imshow((x * s).permute((1, 2, 0)) / 5 + 0.5)
        axarr[2].axis('off')
        f.suptitle("{} , x: {:.2f} , y: {:.2f} , sigma: {:.2f}".format(classes[y.item()], params[0].item(),
                                                                       params[1].item(), params[2].item()))
        plt.show()

    def add_train_losses_to_writer(self, loss_dict, iteration):
        for k, v in loss_dict.items():
            self.args.writer.add_scalar('Train/{}_Iter'.format(k), v, iteration)

    def add_eval_results_to_writer(self, eval_results, partition='Val', iteration=None, epoch=None):
        if (iteration is None) and (epoch is None):
            raise ValueError('One of the iteration or epoch values should be not None.')
        if iteration is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Iter'.format(partition, k), v, iteration)

        elif epoch is not None:
            for k, v in eval_results.items():
                self.args.writer.add_scalar('{}/{}_Epoch'.format(partition, k), v, epoch)

    @staticmethod
    def smoothness_loss(masks, power=2, border_penalty=0.3):
        x_loss = torch.sum((torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power)
        y_loss = torch.sum((torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power)
        if border_penalty > 0:
            border = float(border_penalty) * torch.sum(
                masks[:, :, -1, :] ** power + masks[:, :, 0, :] ** power + masks[:, :, :, -1] ** power +
                masks[:, :, :, 0] ** power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * masks.size(0))
