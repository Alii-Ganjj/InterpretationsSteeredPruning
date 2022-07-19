import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from Models.architectures import encoder_resnet, UpSampleBlock
from Models.ResNetWithGateAEM import RealTimeSaliencyModel, RealTimeSaliencyRBF
from collections import OrderedDict
from Models.sampler import RandomBernoulliSampler
from Models.ResNetWithGateAEM import ResNet
from Models.MobileNet import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np


class BlackBoxModel(nn.Module):
    def __init__(self, args):
        super(BlackBoxModel, self).__init__()
        self.args = args
        self.model = None
        self.build_network()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=[int(0.5 * self.args.num_epochs),
                                                                    int(0.75 * self.args.num_epochs)],
                                                        gamma=0.1)
        self.class_loss = nn.CrossEntropyLoss()

    def build_network(self):
        if self.args.model_type == 'resnet-gate':
            self.model = ResNet(depth=self.args.depth, gate_flag=True)
        elif self.args.model_type == 'MobileNetV2':
            self.model = MobileNetV2()
        else:
            raise ValueError("Blackbox model's architecture should be either ResNet or MobileNet")

        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

    def training_step(self, x_img, y, iteration):
        self.model.train()
        y_pred = self.model(x_img)
        loss = self.class_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {'loss': loss}
        self.add_train_losses_to_writer(loss_dict, iteration)
        return loss_dict

    def eval_model(self, test_dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
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

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('#' * 30)
        logging.warning("loading pretrained blackbox model's checkpoint from: {}".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        first_key = list(checkpoint['model'].keys())[0]
        if ('module' in first_key) and ('resnet' in first_key):  # Old DataParallel blackbox Models trained.
            new_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k.replace('resnet', 'model')  # remove `module.`
                new_dict[name] = v

            if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
                self.load_state_dict(new_dict)
                return

            new_dict2 = OrderedDict()
            for k, v in new_dict.items():
                name = k.replace('module.', '')  # remove `module.`
                new_dict2[name] = v
            self.load_state_dict(new_dict2)

        elif ('module' in first_key) and ('model' in first_key):  # New DataParallel blackbox Models trained. (self.resnet -> self.model)
            if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
                self.load_state_dict(checkpoint['model'])
                return
            new_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_dict[name] = v
            self.load_state_dict(new_dict)

        else:
            self.load_state_dict(checkpoint['model'])

    def forward(self, x):
        return self.model(x)


class Selector(nn.Module):
    def __init__(self, args):
        super(Selector, self).__init__()
        self.args = args
        self.resnet = encoder_resnet()
        self.fix_encoder = self.args.fix_encoder

        if self.args.load_saliency_encoder_from_checkpoint:
            self.resnet.load_checkpoint(self.args.pretrained_model_dir)

        self.uplayer4 = UpSampleBlock(in_channels=512, out_channels=256, passthrough_channels=256)
        self.uplayer3 = UpSampleBlock(in_channels=256, out_channels=128, passthrough_channels=128)
        self.uplayer2 = UpSampleBlock(in_channels=128, out_channels=64, passthrough_channels=64)

        self.embedding = nn.Embedding(self.args.num_classes, 512)
        self.saliency_chans = nn.Conv2d(64, 1, kernel_size=1, bias=False)

    def forward(self, x, labels):
        scale1, scale2, scale3, scale4 = self.resnet(x)

        if self.fix_encoder:
            scale1, scale2, scale3, scale4 = scale1.detach(), scale2.detach(), scale3.detach(), scale4.detach()

        em = torch.squeeze(self.embedding(labels.view(-1, 1)), 1)
        act = torch.sum(scale4 * em.view(-1, 512, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4 * th

        upsample3 = self.uplayer4(scale4, scale3)
        upsample2 = self.uplayer3(upsample3, scale2)
        upsample1 = self.uplayer2(upsample2, scale1)

        saliency_chans = self.saliency_chans(upsample1)
        return saliency_chans

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('loading pretrained checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if 'selector' in k:
                name = k.replace('selector.', '')
                if 'module' in name:
                    name = name.replace('module.', '')
                new_dict[name] = v

        self.load_state_dict(new_dict)


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        if self.args.model_type == 'resnet-gate':
            self.resnet = ResNet(self.args.depth, gate_flag=True)
        elif self.args.model_type == 'MobileNetV2':
            """self.resnet does not necessary mean that the trained model has resnet architecture. I used the name
            resnet in the initial implementations and the previous checkpoints. We will revise this naming later"""
            self.resnet = MobileNetV2()
        else:
            raise ValueError
        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):
            self.resnet = nn.DataParallel(self.resnet, device_ids=self.args.gpu_ids)

    def forward(self, x):
        return self.resnet(x)

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('######## Loading Pretrained Predictor ########')
        logging.warning('loading pretrained checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        new_dict = OrderedDict()
        if (len(self.args.gpu_ids) > 1) and (torch.cuda.is_available()):  # Data Parallel mode on GPUs.
            for k, v in checkpoint['model'].items():
                if not ('predictor' in k):
                    continue
                name = k.replace('predictor.resnet.', '')
                new_dict[name] = v
            self.resnet.load_state_dict(new_dict)
        else:
            for k, v in checkpoint['model'].items():
                if not ('predictor' in k):
                    continue
                if 'predictor.resnet.module.' in k:
                    name = k.replace('predictor.resnet.module.', '')
                else:
                    name = k.replace('predictor.resnet.', '')
                new_dict[name] = v
            self.resnet.load_state_dict(new_dict)


class TrainPredictor(nn.Module):
    def __init__(self, args):
        super(TrainPredictor, self).__init__()
        self.args = args
        self.predictor = Predictor(self.args)
        self.predictor.to(self.args.device)

        self.bb_model = BlackBoxModel(self.args)
        self.bb_model.load_checkpoint(self.args.pretrained_model_dir)
        self.bb_model.to(self.args.device)

        self.loss_fn = F.kl_div
        self.optimizer = optim.Adam(self.predictor.parameters(),
                                    lr=self.args.lr,
                                    betas=self.args.betas,
                                    weight_decay=self.args.weight_decay)
        if self.args.hard_sample and (self.args.selector_type == 'real-x'):  # REAL-X implementation.
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
            probs = self.calculate_rbf(n, h, w)
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
                    probs = self.calculate_rbf(n, h, w)
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
        xy = (torch.rand((n, 2)) * 2. - 1.) * 14. + 16.
        sigma = torch.rand(n) * (self.args.rbf_sig_end - self.args.rbf_sig_start) + self.args.rbf_sig_start
        maps = []
        for i in range(n):
            x_c, y_c = self.coordinate_arrays(h, w)
            map = (1 / (2 * torch.tensor(np.pi) * (sigma[i] ** 2))) * \
                  torch.exp((-1. / (2 * (sigma[i] ** 2))) * (((x_c - xy[i, 0]) ** 2) + ((y_c - xy[i, 1]) ** 2)))
            maps.append(map.unsqueeze(0) / map.detach().max())

        out_maps = torch.stack(maps)
        return out_maps

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


class TrainSelector(nn.Module):
    """
    Implementation of training of our selector model. It also includes implementation of the selector training of REAL-X
    presented in 'https://proceedings.mlr.press/v130/jethani21a/jethani21a.pdf'.
    """
    def __init__(self, args):
        super(TrainSelector, self).__init__()
        self.args = args
        self.lamda = self.args.lamda
        if self.args.selector_type == 'real-x':
            self.selector = RealTimeSaliencyModel(self.args)
        elif self.args.selector_type == 'rbf':
            self.selector = RealTimeSaliencyRBF(self.args)
        else:
            raise ValueError
        self.selector.to(self.args.device)

        self.bb_model = BlackBoxModel(self.args)
        self.bb_model.load_checkpoint(self.args.pretrained_model_dir)
        self.bb_model.to(self.args.device)

        self.predictor = Predictor(self.args)
        self.predictor.load_checkpoint(self.args.pretrained_predictor_dir)
        self.predictor.to(self.args.device)

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

        if self.args.selector_type == 'rbf':
            sel_prob, params = self.selector(x, y)
            params = params.squeeze()
        elif self.args.selector_type == 'real-x':
            sel_prob = self.selector(x, y)
            params = [None] * sel_prob.shape[0]
        else:
            raise ValueError
        sampler = RelaxedBernoulli(temperature=self.args.temperature, probs=sel_prob)
        s_relaxed = sampler.rsample()
        if self.args.hard_sample:
            s = (s_relaxed > 0.5).detach().float().to(self.args.device) - s_relaxed.detach() + s_relaxed
        else:
            s = s_relaxed

        # if iteration % 50 == 0:  # You can uncomment this line to visualize the samples during training.
        #     classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        #     n = 0
        #     self.visualize_sample(s[n], x[n], y[n], params[n])
            # plt.imshow(x[0].permute((1, 2, 0)) / 5. + 0.5)
            # plt.title(classes[y[0]])
            # plt.show()
            #
            # plt.imshow(s[0].detach().squeeze(), cmap='gray')
            # plt.show()
            #
            # plt.imshow((s[0] * x[0]).detach().squeeze().permute((1, 2, 0)) / 5. + 0.5)
            # plt.show()

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
        R_s_approx = torch.mean((s_flat + 0.0005) ** self.args.area_loss_power, 1)

        s_loss = p_f_s + self.lamda * R_s_approx
        s_loss = s_loss.mean()

        if self.args.smoothness_loss_coeff != 0.:
            smooth_loss = self.smoothness_loss(s_relaxed)
            s_loss = s_loss + self.args.smoothness_loss_coeff * smooth_loss # Maybe we can also try smoothness loss on s.
            if iteration % 10 == 0:
                logging.warning('iter: {}, kl: {:.4f}, Rs: {:.4f}, smooth: {:.4f}'.format(iteration,
                                                                                          p_f_s.mean().item(),
                                                                                          R_s_approx.mean().item(),
                                                                                          smooth_loss.item()))
                log_dict = {'kl': p_f_s.mean().item(), 'Rs': R_s_approx.mean().item(), 'smooth': smooth_loss.item()}
                for k, v in log_dict.items():
                    self.args.writer.add_scalar('Train/{}_Iter'.format(k), v, iteration)

        else:
            if iteration % 10 == 0:
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
                if self.args.selector_type == 'rbf':
                    sel_prob, params = self.selector(x, y)
                    params = params.squeeze()
                elif self.args.selector_type == 'real-x':
                    sel_prob = self.selector(x, y)
                    params = [None] * sel_prob.shape[0]
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
    def visualize_sample(s, x, y, params=None):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if params is None:
            s, x, y = s.detach(), x.detach(), y.detach()
        else:
            s, x, y, params = s.detach(), x.detach(), y.detach(), params.detach()
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x.permute((1, 2, 0)) / 5 + 0.5)
        axarr[0].axis('off')
        axarr[1].imshow(s[0], cmap='gray')
        axarr[1].axis('off')
        axarr[2].imshow((x * s).permute((1, 2, 0)) / 5 + 0.5)
        axarr[2].axis('off')
        if params is None:
            f.suptitle("{}".format(classes[y]))
        else:
            f.suptitle("{} , x: {:.2f} , y: {:.2f} , sigma: {:.2f}".format(classes[y], params[0].item(),
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


def image_show(img, name):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(name)
    plt.show()
