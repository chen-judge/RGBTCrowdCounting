import h5py
import torch
import shutil
import os
import numpy as np


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            

def save_checkpoint(state, mae_is_best, mse_is_best, task_id, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(task_id, filename))

    if mae_is_best:
        shutil.copyfile(os.path.join(task_id, filename), os.path.join(task_id, 'model_mae_best.pth.tar'))
    if mse_is_best:
        shutil.copyfile(os.path.join(task_id, filename), os.path.join(task_id, 'model_mse_best.pth.tar'))


def cal_para(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        # print "stucture of layer: " + str(list(i.size()))
        for j in i.size():
            l *= j
        # print "para in this layer: " + str(l)
        k = k + l
    print ("the amount of para: " + str(k))
    return str(k)


