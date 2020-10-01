import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import save_checkpoint, AverageMeter, accuracy, countnonZeroWeights
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet_models import group_lasso_ResNet

parser = argparse.ArgumentParser(description='ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='group_lasso_ResNet32', type=str,
                    help='name of experiment')
parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                    help='whether to use tensorboard (default: True)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--lamba', type=float, default=0.001,
                    help='Coefficient for the L0 term.')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
parser.add_argument('--dataset', choices=['c10', 'c100'], default='c10')
parser.add_argument('--epoch_drop', nargs='*', type=int, default=(60, 120, 160))
parser.add_argument('--multi_gpu', action='store_true')
parser.set_defaults(tensorboard=True)

#declare variables
best_prec1 = 100
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []

def main():
    #set variables global
    global args, best_prec1, writer, time_acc, total_steps, exp_flops, exp_l0, param_num
    
    #parse the arguments
    args = parser.parse_args()
    log_dir_net = args.name

    #set up tensorboard
    if args.tensorboard:
        # used for logging to TensorBoard
        from tensorboardX import SummaryWriter
        directory = 'logs/{}/{}'.format(log_dir_net, args.name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
        writer = SummaryWriter(directory)
    if args.dataset == 'c10':
    	num_classes = 10
    elif args.dataset == 'c100':
    	num_classes = 100
    #crate model
    model = group_lasso_ResNet(num_classes = num_classes, weight_decay = args.weight_decay,
    	lamba = args.lamba)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    param_num = sum([p.data.nelement() for p in model.parameters()])

    # for training on multiple GPUs. 
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

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
            time_acc = checkpoint['time_acc']
            exp_flops = checkpoint['exp_flops']
            exp_l0 = checkpoint['exp_l0']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            total_steps, exp_flops, exp_l0 = 0, [], []

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #create loss function
    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    # define loss function (criterion) and optimizer
    def loss_function(output, target_var, model):
        loss = loglike(output, target_var)
        reg = model.regularization() if not args.multi_gpu else model.module.regularization()
        total_loss = loss + reg
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = True)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_drop, gamma=args.lr_decay_ratio)

    for epoch in range(args.start_epoch, args.epochs):
        time_glob = time.time()

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        prec1_tr = train(train_loader, model, loss_function, optimizer, lr_scheduler, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, loss_function, epoch)
        time_ep = time.time() - time_glob
        time_acc.append((time_ep+time_acc[-1][0], prec1_tr, prec1))

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        state = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'curr_prec1': prec1,
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps,
            'time_acc': time_acc,
            'exp_flops': exp_flops,
            'exp_l0': exp_l0
        }
        save_checkpoint(state,is_best, args.name)

    print('Best error: ', best_prec1)
    if args.tensorboard:
        writer.close()

def train(train_loader, model, criterion, optimizer, lr_schedule, epoch):
    """Train for one epoch on the training set"""
    global total_steps, exp_flops, exp_l0, args, writer, param_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    lr_schedule.step(epoch=epoch)
    if writer is not None:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(100 - prec1.item(), input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
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
    """Perform validation on the validation set"""
    global args, writer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input_ = input_.cuda()

        with torch.no_grad():
            input_var = torch.autograd.Variable(input_)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, model)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input_.size(0))
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