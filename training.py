import torch
import logging
import copy
import os


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

                # if (itr + 1) % args.testing_iter == 0:  # Uncomment this part for the selector training stage for more evaluations.
                #     eval_val = model.eval_model(val_dataloader)
                #     eval_test = model.eval_model(test_dataloader)
                #     model.add_eval_results_to_writer(eval_val, 'Val', iteration=iteration)
                #     model.add_eval_results_to_writer(eval_test, 'Test', iteration=iteration)
                #     logging.warning(prepare_logging_message(epoch_loss.item(), eval_val, eval_test, epoch=epoch,
                #                                             num_epochs=args.num_epochs, itr=itr + 1,
                #                                             num_iter=len(train_dataloader)))
                #     logging.warning('Saving model epoch #{}, Itr: {}'.format(epoch, itr + 1))
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
                if (epoch <= 2) and (iteration % 2000 == 0):
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
