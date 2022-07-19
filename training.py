import torch
import logging
import math
import copy
import os
from tqdm import tqdm
import torch.nn as nn
from utils import display_structure, loss_fn_kd, loss_label_smoothing, display_factor, display_structure_hyper, \
    LabelSmoothingLoss


def prepare_logging_message_old(loss, eval_val=None, epoch=None, num_epochs=None, itr=None, num_iter=None):
    if eval_val is None:  # At the end of iter without evaluating on val/test set.
        log = 'Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f}'.format(epoch, num_epochs, itr, num_iter, loss)
    else:
        if itr is None:
            log = 'Epoch: {}/{} \t Epoch_Loss: {:.4f}\n'.format(epoch, num_epochs, loss)
            for k, v in eval_val.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
        else:
            log = 'Epoch: {}/{} \t Iter: {}/{}\n'.format(epoch, num_epochs, itr, num_iter)
            for k, v in eval_val.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
    return log


def prepare_logging_message(loss, eval_val=None, eval_test=None, epoch=None, num_epochs=None, itr=None, num_iter=None):
    if eval_val is None:  # At the end of iter without evaluating on val/test set.
        log = 'Epoch: {}/{} \t Iter: {}/{} \t Loss: {:.4f}'.format(epoch, num_epochs, itr, num_iter, loss)

    else:
        if itr is None:
            log = 'Epoch: {}/{} \t Epoch_Loss: {:.4f}\n'.format(epoch, num_epochs, loss)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
        else:
            log = 'Epoch: {}/{} \t Iter: {}/{}\n'.format(epoch, num_epochs, itr, num_iter)
            for k, v in eval_val.items():
                log += 'val_{}: {:.4f}\n'.format(k, v)
            for k, v in eval_test.items():
                log += 'test_{}: {:.4f}\n'.format(k, v)
    return log


def train_model(model, datamodule, args):
    train_dataloader, val_dataloader, test_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader(),\
                                                        datamodule.test_dataloader()
    checkpoint = {'model': None, 'val': {'loss': 1e9, 'acc': 0., 'kl_loss': 1e9},
                  'test': {'loss': 1e9, 'acc': 0., 'kl_loss': 1e9}, 'iter': 0}
    epoch, iteration = 0, 0
    model.train()
    if not args.debug:
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss, total = 0., 0.
            for itr, batch in enumerate(train_dataloader, 0):
                iteration += 1
                model.train()
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                loss_dict = model.training_step(x, y, iteration)
                epoch_loss += loss_dict['loss'] * y.shape[0]
                total += y.shape[0]
                if (itr + 1) % args.logging_freq == 0:
                    logging.warning(prepare_logging_message(loss_dict['loss'].item(), epoch=epoch,
                                                            num_epochs=args.num_epochs, itr=itr + 1,
                                                            num_iter=len(train_dataloader)))

                # if (itr + 1) % args.testing_iter == 0: # Uncomment this part for the selector training stage for more evaluations.
                #     eval_val = model.eval_model(val_dataloader)
                #     eval_test = model.eval_model(test_dataloader)
                #     model.add_eval_results_to_writer(eval_val, 'Val', iteration=iteration)
                #     model.add_eval_results_to_writer(eval_test, 'Test', iteration=iteration)
                #     logging.warning(prepare_logging_message(epoch_loss.item(), eval_val, eval_test, epoch=epoch,
                #                                             num_epochs=args.num_epochs, itr=itr + 1,
                #                                             num_iter=len(train_dataloader)))
                #     logging.warning('Saving model epoch #{}, Itr: {}'.format(epoch, itr))
                #     save_model(model, iteration, args)
                #     checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, iteration)
                # if epoch == 1 and iteration % 100 == 0:
                #     logging.warning('Saving model epoch #{}, Itr: {}'.format(epoch, itr))
                #     save_model(model, iteration, args)

            epoch_loss = epoch_loss / total
            args.writer.add_scalar('Train/loss_Epoch', epoch_loss, epoch)
            if epoch % args.testing_epoch == 0:
                logging.warning('Evaluating Network at the end of epoch #{}:'.format(epoch))
                eval_val = model.eval_model(val_dataloader)
                eval_test = model.eval_model(test_dataloader)
                model.add_eval_results_to_writer(eval_val, 'Val', epoch=epoch)
                model.add_eval_results_to_writer(eval_test, 'Test', epoch=epoch)
                logging.warning(prepare_logging_message(epoch_loss.item(), eval_val, eval_test, epoch=epoch,
                                                        num_epochs=args.num_epochs))
                checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, iteration)

            if (epoch % args.saving_epoch) == 0:
                logging.warning('Saving model at the end of epoch #{}'.format(epoch))
                save_model(model, iteration, args)

            if model.scheduler:
                model.scheduler.step()
        return checkpoint

    else:
        model.train()
        train_batch = next(iter(train_dataloader))
        x, y = train_batch[0].to(args.device), train_batch[1].to(args.device)
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            loss_dict = model.training_step(x, y, iteration)
            epoch_loss = loss_dict['loss']
            eval_val = model.eval_model(val_dataloader)
            eval_test = model.eval_model(test_dataloader)
            model.add_eval_results_to_writer(eval_test, 'Test', epoch=epoch)
            logging.warning(prepare_logging_message(epoch_loss.item(), eval_val, eval_test, epoch=epoch,
                                                    num_epochs=args.num_epochs))
            checkpoint = update_checkpoint(checkpoint, model, eval_val, eval_test, args, epoch)
        return checkpoint


def train_model_no_val(model, datamodule, args):
    train_dataloader, test_dataloader = datamodule.train_dataloader(), datamodule.test_dataloader()
    checkpoint = {'model': None, 'test': {'loss': 1e9, 'acc': 0.}, 'iter': 0}
    epoch, iteration = 0, 0
    model.train()
    if not args.debug:
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss, total = 0., 0.
            for itr, batch in enumerate(train_dataloader, 0):
                model.train()
                x, y = batch[0].to(args.device), batch[1].to(args.device)
                loss_dict = model.training_step(x, y, iteration)
                epoch_loss += loss_dict['loss'] * y.shape[0]
                total += y.shape[0]
                iteration += 1
                if (itr + 1) % args.logging_freq == 0:
                    logging.warning(prepare_logging_message_old(loss_dict['loss'].item(),
                                                                epoch=epoch,
                                                                num_epochs=args.num_epochs,
                                                                itr=itr + 1,
                                                                num_iter=len(train_dataloader)))
                if (epoch <= 2) and (iteration % 100 == 0):
                    logging.warning('Saving model at the end of iteration #{}'.format(iteration))
                    save_model(model, iteration, args)

            epoch_loss = epoch_loss / total
            args.writer.add_scalar('Train/loss_Epoch', epoch_loss, epoch)
            if epoch % args.testing_epoch == 0:
                logging.warning('Evaluating Network at the end of epoch #{}:'.format(epoch))
                eval_test = model.eval_model(test_dataloader)
                model.add_eval_results_to_writer(eval_test, 'Test', epoch=epoch)
                logging.warning(prepare_logging_message_old(epoch_loss.item(),
                                                            eval_test,
                                                            epoch=epoch,
                                                            num_epochs=args.num_epochs))
                checkpoint = update_checkpoint_old(checkpoint, model, eval_test, iteration, args.metric)

            if model.scheduler:
                model.scheduler.step()
        return checkpoint

    else:
        model.train()
        train_batch = next(iter(train_dataloader))
        x, y = train_batch[0].to(args.device), train_batch[1].to(args.device)
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            loss_dict = model.training_step(x, y, iteration)
            epoch_loss = loss_dict['loss']
            eval_test = model.eval_model(test_dataloader)
            model.add_eval_results_to_writer(eval_test, 'Test', epoch=epoch)
            logging.warning(prepare_logging_message_old(epoch_loss.item(),
                                                        eval_test,
                                                        epoch=epoch,
                                                        num_epochs=args.num_epochs))
            checkpoint = update_checkpoint_old(checkpoint, model, eval_test, epoch, args.metric)
        return checkpoint


def update_checkpoint_old(curr_checkpoint_dict, model, eval_test, iteration, metric='acc'):
    if metric == 'acc':
        if curr_checkpoint_dict['test'][metric] > eval_test[metric]:
            return curr_checkpoint_dict
    elif metric == 'loss':
        if curr_checkpoint_dict['test'][metric] < eval_test[metric]:
            return curr_checkpoint_dict
    else:
        raise NotImplementedError
    updated_dict = dict(curr_checkpoint_dict)
    updated_dict['model'] = copy.deepcopy(model.state_dict())
    updated_dict['test'] = dict(eval_test)
    updated_dict['iter'] = iteration
    return updated_dict


def save_checkpoint(checkpoint, args):
    checkpoint_name = 'checkpoint_iter_{}.pth'.format(checkpoint['iter'])
    file_name = os.path.join(args.checkpoint_dir, checkpoint_name)
    logging.warning('Saving Checkpoint: {}'.format(file_name))
    torch.save(checkpoint, file_name)


def save_model(model, iteration, args):
    checkpoint_name = 'checkpoint_iter_{}.pth'.format(iteration)
    file_name = os.path.join(args.checkpoint_dir, checkpoint_name)
    logging.warning('Saving Checkpoint: {}'.format(file_name))
    checkpoint = {'model': copy.deepcopy(model.state_dict()), 'iter': iteration}
    torch.save(checkpoint, file_name)


def update_checkpoint(curr_checkpoint_dict, model, eval_val, eval_test, args, iteration):
    if args.metric == 'loss' or args.metric == 'kl_loss':
        if eval_val[args.metric] > curr_checkpoint_dict['val'][args.metric]:
            return curr_checkpoint_dict
    elif eval_val[args.metric] < curr_checkpoint_dict['val'][args.metric]:
        return curr_checkpoint_dict

    updated_dict = dict(curr_checkpoint_dict)
    updated_dict['model'] = copy.deepcopy(model.state_dict())
    updated_dict['val'] = dict(eval_val)
    updated_dict['test'] = dict(eval_test)
    updated_dict['iter'] = iteration
    return updated_dict


def train_IGP(epoch, net, hyper_net, criterion, trainloader, optimizer, res_con, args, txt_name=None):
    tqdm_loader = tqdm(trainloader)

    net.eval()
    net.freeze_weights()

    hyper_net.train()

    train_loss = 0
    resource_loss = 0
    mse_losses = 0
    c_losses = 0

    correct = 0
    total = 0

    for batch_idx, (masks, ori_inputs, targets) in enumerate(tqdm_loader):
        masks, ori_inputs, targets = masks.cuda(), ori_inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        vector = hyper_net()

        net.resnet.set_vritual_gate(vector)
        pred_masks, pred = net(ori_inputs, targets, pred=True)

        # mse_loss = (pred_masks - masks).pow(2).sum(dim=0).mean()
        mse_loss = nn.MSELoss()(pred_masks.view(pred_masks.size(0), -1), masks.view(masks.size(0), -1))
        res_loss = res_con(hyper_net.resource_output())

        if args.c_loss:
            c_loss = criterion(pred, targets)
        else:
            c_loss = 0

        loss = c_loss + args.mse_w * mse_loss + res_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mse_losses += mse_loss.item()
        resource_loss += res_loss.item()

        if args.c_loss:
            c_losses += c_loss.item()
        else:
            c_losses += 0

        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch: %d Loss: %.3f MSE Loss: %.3f Res Loss: %.3f Clf Loss: %.3f  | Acc: %.3f%% (%d/%d)'
          % (epoch, train_loss / len(trainloader), mse_losses / len(trainloader),
             resource_loss / len(trainloader), c_losses / len(trainloader), 100. * correct / total, correct, total))
    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(mse_losses / len(trainloader)) + ' ' + str(resource_loss / len(tqdm_loader)) + ' ' + str(
            c_losses / len(tqdm_loader)) + ' ' + str(train_loss / len(tqdm_loader))

        file_txt.write(contents + ' \n')
        file_txt.close()


def train_IGP_gau(epoch, net, hyper_net, criterion, trainloader, optimizer, res_con, args, txt_name=None):
    tqdm_loader = tqdm(trainloader)

    net.eval()
    net.freeze_weights()

    hyper_net.train()

    train_loss = 0
    resource_loss = 0
    mse_losses = 0
    c_losses = 0

    correct = 0
    total = 0

    for batch_idx, (masks, ori_inputs, targets) in enumerate(tqdm_loader):
        masks, ori_inputs, targets = masks.cuda(), ori_inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        vector = hyper_net()

        net.resnet.set_vritual_gate(vector)

        if args.gau:
            _, pred_masks, pred = net(ori_inputs, targets, pred=True)
            pred_masks = pred_masks.squeeze()
        else:
            pred_masks, pred = net(ori_inputs, targets, pred=True)

        mse_loss = nn.MSELoss()(pred_masks, masks)
        res_loss = res_con(hyper_net.resource_output())

        if args.c_loss:
            c_loss = criterion(pred, targets)
        else:
            c_loss = 0

        loss = c_loss + args.mse_w * mse_loss + res_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        resource_loss += res_loss.item()

        if args.c_loss:
            c_losses += c_loss.item()
        else:
            c_losses += 0

        mse_losses += mse_loss.item()

        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch: %d Loss: %.3f MSE Loss: %.3f Res Loss: %.3f Clf Loss: %.3f  | Acc: %.3f%% (%d/%d)'
          % (epoch, train_loss / len(trainloader), mse_losses / len(trainloader),
             resource_loss / len(trainloader), c_losses / len(trainloader), 100. * correct / total, correct, total))
    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(mse_losses / len(trainloader)) + ' ' + str(resource_loss / len(tqdm_loader)) + ' ' + str(
            c_losses / len(tqdm_loader)) + ' ' + str(train_loss / len(tqdm_loader))

        file_txt.write(contents + ' \n')
        file_txt.close()


def valid(epoch, net, testloader, best_acc, hyper_net=None, model_string=None, txt_name=None):
    tqdm_loader = tqdm(testloader)
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    if hyper_net is not None:
        hyper_net.eval()
        vector = hyper_net()
        net.set_vritual_gate(vector)
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if hyper_net is not None:

                outputs = net(inputs)
                loss = criterion(outputs, targets)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    is_best = False
    if hyper_net is not None:
        if epoch > 100:
            if acc > best_acc:
                best_acc = acc
                is_best = True
        else:
            best_acc = 0
    else:
        if acc > best_acc:
            best_acc = acc
            is_best = True
    if model_string is not None:

        if is_best:
            print('Saving..')
            if hyper_net is not None:

                state = {
                    'net': net.state_dict(),
                    'hyper_net': hyper_net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector': vector
                    # 'gpinfo':gplist,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    # 'gpinfo':gplist,
                }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/%s.pth.tar' % (model_string))

    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
          % (test_loss / len(testloader), 100. * correct / total, correct, total, best_acc))

    if txt_name is not None:
        # print(txtdir + txt_name)
        file_txt = open(txt_name, 'a')

        contents = str(correct / total) + ' ' + str(test_loss / len(testloader))

        file_txt.write(contents + ' \n')
        file_txt.close()

    return best_acc


def retrain(epoch, net, criterion, trainloader, optimizer, smooth=True, scheduler=None, alpha=0.5):
    tqdm_loader = tqdm(trainloader)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = alpha

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        if scheduler is not None:
            scheduler.step()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        if smooth:
            loss_smooth = LabelSmoothingLoss(classes=10, smoothing=0.1)(outputs, targets)
            loss_c = criterion(outputs, targets)
            loss = alpha * loss_smooth + (1 - alpha) * loss_c
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (epoch, train_loss / len(trainloader), 100. * correct / total, correct, total))
