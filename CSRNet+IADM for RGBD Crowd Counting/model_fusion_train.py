import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import argparse
import json
import dataset
import time
import logging
import random

from CSRNet_IADM import FusionCSRNet

from utils import save_checkpoint
from utils import cal_para


parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--train_json', '--train', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--test_json', '--test', metavar='TEST',
                    help='path to test json')
parser.add_argument('--checkpoint', '-c', metavar='CHECKPOINT', default=None,type=str,
                    help='path to the checkpoint')
parser.add_argument('--lr', default=None, type=float,
                    help='learning rate')
parser.add_argument('--save', metavar='SAVE', type=str,
                    help='save path')

global args
args = parser.parse_args()

if os.path.isdir(args.save) is False:
    os.makedirs(args.save)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%d-%H:%M",
                    handlers=[
                            logging.StreamHandler()
                        ])
logger = logging.getLogger()
fh = logging.FileHandler("{0}/{1}.log".format(args.save, 'log'), mode='w')
fh.setFormatter(logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="%d-%H:%M"))
logger.addHandler(fh)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    
    global args, mae_best_prec1, mse_best_prec1
    
    mae_best_prec1 = 1e6
    mse_best_prec1 = 1e6

    args.original_lr = args.lr
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.workers = 12

    args.seed = random.randint(0, 9999999)
    setup_seed(args.seed)

    args.print_freq = 1500
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    model = FusionCSRNet()

    logger.info(model)
    logger.info('Parameters: '+cal_para(model))
    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            mae_best_prec1 = checkpoint['mae_best_prec1']
            mse_best_prec1 = checkpoint['mse_best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, model, criterion, optimizer, epoch)
        mae_prec1, mse_prec1 = test(val_list, model)

        mae_is_best = mae_prec1 < mae_best_prec1
        mae_best_prec1 = min(mae_prec1, mae_best_prec1)
        mse_is_best = mse_prec1 < mse_best_prec1
        mse_best_prec1 = min(mse_prec1, mse_best_prec1)
        if mae_is_best:
            logger.info(' *** Best MAE *** ')
        if mse_is_best:
            logger.info(' *** Best MSE *** ')
        logger.info (' * best MAE {mae:.2f}  * best MSE {mse:.2f}'
              .format(mae=mae_best_prec1, mse=mse_best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.checkpoint,
            'state_dict': model.state_dict(),
            'mae_best_prec1': mae_best_prec1,
            'mse_best_prec1': mse_best_prec1,
            'optimizer': optimizer.state_dict(),
        }, mae_is_best, mse_is_best, args.save)


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            train=True, seen=model.seen,
                            num_workers=args.workers,),
        num_workers=args.workers,
        shuffle=True,
        batch_size=1)
    logger.info('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    logger.info(args.save)

    model.train()
    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        assert type(data) == list
        data[0] = Variable(data[0].cuda()).type(torch.FloatTensor).cuda()
        data[1] = Variable(data[1].cuda()).type(torch.FloatTensor).cuda()

        output = model(data)

        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        loss = criterion(output, target)
        losses.update(loss.item(), data[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == args.print_freq-1:
            log_text = ('Epoch: [{0}][{1}/{2}]  '
                        'Time {batch_time.avg:.3f}  '
                        'Data {data_time.avg:.3f}  '
                        'Loss {loss.avg:.4f} '
                .format(epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            logger.info(log_text)


def test(val_list, model):
    logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            train=False),
        shuffle=False,
        batch_size=1)

    model.eval()

    mae = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]

    for i, (data, target) in enumerate(test_loader):
        data[0] = Variable(data[0]).type(torch.FloatTensor).cuda()
        data[1] = Variable(data[1]).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            output = model(data)

        for L in range(4):
            abs_error, square_error = evaluation(output, target, L)
            mae[L] += abs_error
            mse[L] += square_error

    N = len(test_loader)
    mae = [m / N for m in mae]
    mse = [torch.sqrt(m / N) for m in mse]

    logger.info (' * MAE0 {mae0:.2f} * MAE1 {mae1:.2f} * MAE2 {mae2:.2f} * MAE3 {mae3:.2f} * MSE {mse:.2f}'
          .format(mae0=mae[0], mae1=mae[1], mae2=mae[2], mae3=mae[3], mse=mse[0]))

    return mae[0], mse[0]


def evaluation(output, target, L=0):
    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    _, _, H, W = target.shape
    for i in range(p):
        for j in range(p):
            # print i, j, (i*H/p,(i+1)*H/p), (j*W/p,(j+1)*W/p)
            output_block = output[:, :, i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[:, :, i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            abs_error += abs(output_block.cpu().data.sum()-target_block.cpu().data.sum().float())
            square_error += (output_block.cpu().data.sum()-target_block.cpu().data.sum().float()).pow(2)
    return abs_error, square_error


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


if __name__ == '__main__':
    main()        