import os
import torch
from torch.autograd import Variable

import argparse
import json
import dataset

from CSRNet_IADM import FusionCSRNet
from utils import cal_para

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('--test_json', '--test', metavar='TEST',
                    help='path to test json')

parser.add_argument('--checkpoint', '-c', metavar='CHECKPOINT', default=None, type=str,
                    help='path to the checkpoint')

global args
args = parser.parse_args()


def main():

    args.workers = 12

    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    model = FusionCSRNet()

    print(model)
    print('Parameters: ' + cal_para(model))
    model = model.cuda()

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    test(val_list, model)


def test(val_list, model):
    print('begin test')
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

    print(' * MAE0 {mae0:.2f} * MAE1 {mae1:.2f} * MAE2 {mae2:.2f} * MAE3 {mae3:.2f} * MSE {mse:.2f}'
                .format(mae0=mae[0], mae1=mae[1], mae2=mae[2], mae3=mae[3], mse=mse[0]))


def evaluation(output, target, L=0):
    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    _, _, H, W = target.shape
    for i in range(p):
        for j in range(p):
            output_block = output[:, :, i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]
            target_block = target[:, :, i * H // p:(i + 1) * H // p, j * W // p:(j + 1) * W // p]
            abs_error += abs(output_block.cpu().data.sum() - target_block.cpu().data.sum().float())
            square_error += (output_block.cpu().data.sum() - target_block.cpu().data.sum().float()).pow(2)
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