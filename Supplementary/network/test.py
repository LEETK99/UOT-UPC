import os
import sys 

# cuda_index = '0'
cuda_index = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../util')
from my_dataset import RealComGANDataset
from test_util_gan import TestFramework
from io_util import *

import argparse

from usspa import USSPA, USSPA_split, USSPA_split_solo


def save_func(path, data, outputs, criterion, loss, time):
    if not os.path.exists(path):
        os.mkdir(path)
    inputs = data[0][0][0].to('cpu').detach().numpy()
    sn_data = data[1][0][0].to('cpu').detach().numpy()
    gts = data[0][1][0].to('cpu').detach().numpy()
    
    x_fake = outputs[0][0].to('cpu').detach().numpy()
    x_fake_2 = outputs[1][0].to('cpu').detach().numpy()
    #point_R_0 = outputs[-1][0][0].to('cpu').detach().numpy()
    #point_m_1 = outputs[-1][1][0].to('cpu').detach().numpy()
    # point_m_2 = outputs[-1][2][0].to('cpu').detach().numpy()
    #point_A = outputs[2][0].to('cpu').detach().numpy()
    #point_A_2 = outputs[3][0].to('cpu').detach().numpy()
    #input_R_point_R_0 = outputs[4][0].to('cpu').detach().numpy()
    #point_R_4 = outputs[-1][2][0].to('cpu').detach().numpy()
    # point_R_5 = outputs[-1][3][0].to('cpu').detach().numpy()
    
    write_point_cloud_as_ply(os.path.join(path, 'input'), inputs)
    write_point_cloud_as_ply(os.path.join(path, 'gts'), gts)
    write_point_cloud_as_ply(os.path.join(path, 'res_2'), x_fake_2)
    
    '''
    write_point_cloud_as_ply(os.path.join(path, 'input'), inputs)
    write_point_cloud_as_ply(os.path.join(path, 'res'), x_fake)
    write_point_cloud_as_ply(os.path.join(path, 'res_2'), x_fake_2)
    write_point_cloud_as_ply(os.path.join(path, 'point_R_0'), point_R_0)
    write_point_cloud_as_ply(os.path.join(path, 'point_A'), point_A)
    write_point_cloud_as_ply(os.path.join(path, 'point_A_2'), point_A_2)
    write_point_cloud_as_ply(os.path.join(path, 'point_R_4'), point_R_4)
    # write_point_cloud_as_ply(os.path.join(path, 'point_R_5'), point_R_5)
    write_point_cloud_as_ply(os.path.join(path, 'input_R_point_R_0'), input_R_point_R_0)
    write_point_cloud_as_ply(os.path.join(path, 'point_m_1'), point_m_1)
    # write_point_cloud_as_ply(os.path.join(path, 'point_m_2'), point_m_2)
    write_point_cloud_as_ply(os.path.join(path, 'gts'), gts)
    write_point_cloud_as_ply(os.path.join(path, 'sn_data'), sn_data)
    '''
    text = ''
    for j, name in enumerate(criterion.loss_name):
        text += '%s: %f, ' % (name, loss[j])
    with open(os.path.join(path, 'loss.log'), 'w') as f:
        f.write(text + '\n')
        f.write('time: %f\n' % time)


def test(args):
    valid_dataset = RealComGANDataset(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
    #net = USSPA(args)

    test_framework = TestFramework(args.log_dir, cuda_index, False, False)
    test_framework._set_dataset(args.lmdb_valid, valid_dataset)
    if args.ver=='split':
        net = USSPA_split(args)
        test_framework._set_net(net, 'USSPA_split')
    else :
        net = USSPA(args)
        test_framework._set_net(net, 'USSPA')
    if args.ver=='split':
        net = USSPA_split(args)
        test_framework._set_net(net, 'USSPA_split')
    elif args.ver=='solo':
        net = USSPA_split_solo(args)
        test_framework._set_net(net, 'USSPA_split_solo')
        print(args.ver)
    else :
        net = USSPA(args)
        test_framework._set_net(net, 'USSPA')
    # test_framework.test(save_func, last_epoch=args.last_epoch, save_index=list(range(100)))
    test_framework.test(save_func, last_epoch=args.last_epoch, save_index=list(range(100)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lmdb_train', default='../../../data/RealComData/realcom_data_train.lmdb')
    parser.add_argument('--lmdb_valid', default='../../../data/RealComData/realcom_data_test.lmdb')
    parser.add_argument('--lmdb_sn', default='../../../data/RealComShapeNetData/shapenet_data.lmdb')
    parser.add_argument('--class_name', default='all', choices=['all', 'chair', 'table', 'trash_bin', 'tv_or_monitor', 'cabinet', 'bookshelf', 'sofa', 'lamp', 'bed', 'tub'])
    parser.add_argument('--log_dir', default='weights/usspa')

    parser.add_argument('--input_pn', type=int, default=2048)
    parser.add_argument('--gt_pn', type=int, default=2048)
    parser.add_argument('--last_epoch', type=int, default=None)
    parser.add_argument('--phi1', type=str ,default='softplus', choices=['linear', 'kl','sofplus'])
    parser.add_argument('--phi2', type=str, default='softplus', choices=['linear', 'kl','sofplus'])
    parser.add_argument('--tau1', default = 100, type = float)
    parser.add_argument('--tau2', default = 100, type = float)
    parser.add_argument('--lmbd', default = 0, type = float)
    parser.add_argument('--ver', default = 'optimal', type = str, help = ['solo', 'split', 'optimal', 'unperfectoptimal', 'consecutive'])
    parser.add_argument('--TC', default = 'T12', type=str, choices = ['T1_bi', 'T2', 'T11', 'T12', 'T22'])
    parser.add_argument('--cost_type', default = 'cd', type=str,  choices=['cd', 'l2', 'mix'])

    args = parser.parse_args()
    #args.log_dir = os.path.join(args.log_dir, args.class_name)
    test(args)
    