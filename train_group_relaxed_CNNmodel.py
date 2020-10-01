import argparse
import shutil
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


from utils import save_checkpoint
from torch.optim import lr_scheduler
from utils import AverageMeter, accuracy, countnonZeroWeights

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from dataloaders import mnist
from ModelCNN_models import group_relaxed_L0ModelCNN, group_relaxed_L1ModelCNN, group_relaxed_L1L2ModelCNN, group_relaxed_SCAD_ModelCNN, group_relaxed_TF1_ModelCNN

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='CGESCNN', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('--lamba', type=float, default=0.1)
parser.add_argument('--epoch_drop', nargs='*', type=int, default=(40, 80, 120, 160))
parser.add_argument('--beta', type=float, default=2.5,
                    help='Coefficient for the L2 difference of weigth and u.')
parser.add_argument('--growth_factor', type=float, default=1.25,
                    help='growth_factor for beta.')
parser.add_argument('--reg', default='l0', type=str,
                    help='regularizer')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
parser.add_argument('--multi_gpu', action='store_true')
parser.set_defaults(tensorboard=True)

best_prec1 = 100
writer = None
total_steps = 0
exp_flops, exp_l0 = [], []

def main():
    global args, best_prec1, writer, total_steps, exp_flops, exp_l0, param_num
    args = parser.parse_args()
    log_dir_net = args.name
    print ('modl:', args.name)
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        directory = 'logs/{}/{}'.format(log_dir_net, args.name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        writer = SummaryWriter(directory)

    train_loader, val_loader, num_classes = mnist(args.batch_size, pm=False)

    if args.reg== 'l0':
        model = group_relaxed_L0ModelCNN(lamba = args.lamba, beta = args.beta)
    elif args.reg == 'l1':
        model = group_relaxed_L1ModelCNN(lamba = args.lamba, beta = args.beta)
    elif args.reg == 'l1l2':
        model = group_relaxed_L1L2ModelCNN(lamba = args.lamba, beta = args.beta)
    elif args.reg == 'SCAD':
        model = group_relaxed_SCAD_ModelCNN(lamba = args.lamba, beta = args.beta)
    elif args.reg == 'TF1':
        model = group_relaxed_TF1_ModelCNN(lamba = args.lamba, beta = args.beta)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), args.lr)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    param_num = sum([p.data.nelement() for p in model.parameters()])

    print('Number of neurons: ', model.count_total_neuron())


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            total_steps = checkpoint['total_steps']
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            total_steps, exp_flops, exp_l0 = 0, [], []
    cudnn.benchmark = True

    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        total_loss = loss + model.regularization()
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_drop, gamma=args.lr_decay_ratio)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, loss_function, optimizer, lr_schedule, epoch)
        #evaluate on validation set
        prec1 = validate(val_loader, model, loss_function, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'curr_prec1': prec1,
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps,
            'exp_flops': exp_flops,
            'exp_l0': exp_l0
        }
        save_checkpoint(state, is_best, args.name)
    print('Best error: ', best_prec1)
    if args.tensorboard:
        writer.close()


def train(train_loader, model, criterion, optimizer, lr_schedule, epoch):
    global total_steps, exp_flops, exp_l0, args, writer, param_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    lr_schedule.step(epoch=epoch)

    end=time.time()

    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time()-end)
        total_steps +=1
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_=input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        #compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(100-prec1.item(), input_.size(0))

        #compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clamp the parameters
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        e_fl, e_l0 = model.get_exp_flops_l0() if not args.multi_gpu else \
            model.module.get_exp_flops_l0()
        exp_flops.append(e_fl)
        exp_l0.append(e_l0)
        if writer is not None:
            writer.add_scalar('stats_comp/exp_flops', e_fl, total_steps)
            writer.add_scalar('stats_comp/exp_l0', e_l0, total_steps)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # input()
        if i % args.print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

    if (epoch+1)%20 == 0:
        for k, layer in enumerate(layers):
            layer.grow_beta(args.growth_factor)

    u_sparsity = model.get_u_sparsity()
    print('sparsity u: ', u_sparsity)

    w_sparsity = model.get_w_sparsity()
    print('sparsity w:', w_sparsity)

    nonzero_weight = countnonZeroWeights(model)
    print('Number of nonzero weights: ', nonzero_weight)

    neuron = model.count_active_neuron()
    print('Number of active neurons: ', neuron)

    reg_neuron = model.count_reg_neuron_sparsity()
    print('Regularized neuron sparsity: ', reg_neuron)

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/err', top1.avg, epoch)
        writer.add_scalar('w_sparsity/epoch', w_sparsity, epoch)
        writer.add_scalar('sparsity', 1-(nonzero_weight/param_num), epoch)
        writer.add_scalar('neuron sparsity', 1-(neuron/model.count_total_neuron()), epoch)
        writer.add_scalar('active neuron', neuron, epoch)
        writer.add_scalar('regularized neuron sparsity', reg_neuron, epoch)

    return top1.avg

def validate(val_loader, model, criterion, epoch):
    global args, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end =time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        #compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(100 - prec1.item(), input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Err@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Err@1 {top1.avg:.3f}'.format(top1=top1))

    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/err', top1.avg, epoch)
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if hasattr(layer, 'qz_loga'):
                mode_z = layer.sample_z(1, sample=0).view(-1)
                writer.add_histogram('mode_z/layer{}'.format(k), mode_z.cpu().data.numpy(), epoch)

    return top1.avg

if __name__ == '__main__':
    main()
