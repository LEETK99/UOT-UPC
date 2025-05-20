import os
import sys 

# cuda_index = '1'
##cuda_index = '1'
#os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index
#print(os.environ["CUDA_VISIBLE_DEVICES"])
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../util')
from my_dataset import RealComGANDataset, RealComGANDataset_motivation_balanced, RealComGANDataset_motivation_unbalanced, RealComGANDataset_motivation_balanced1, PCNGANDataset
from train_util_gan import TrainFramework
from loss import TrainLoss, TestLoss

import argparse

from usspa import USSPA_split

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.motivation:
        train_dataset = RealComGANDataset_motivation_balanced1(args.lmdb_train, args.lmdb_sn, args.input_pn, args.gt_pn, args.target_ratio, args.class_name)
        valid_dataset = RealComGANDataset_motivation_balanced1(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, [1, 1], args.class_name)   
    elif args.pcn :
        args.lmdb_train = '../../../data/PCN/train'
        args.lmdb_valid = '../../../data/PCN/test'
        train_dataset = PCNGANDataset(args.lmdb_train, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
        valid_dataset = PCNGANDataset(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
    else : 
        train_dataset = RealComGANDataset(args.lmdb_train, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
        valid_dataset = RealComGANDataset(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)

    tf = TrainFramework(args, os.environ["CUDA_VISIBLE_DEVICES"],)
    tf._set_dataset(args.lmdb_train, args.lmdb_valid, train_dataset, valid_dataset)


    net = USSPA_split(args)
    tf._set_net(net, 'USSPA_split')
    train_loss, test_loss = TrainLoss(args), TestLoss(args)
    tf._set_loss(train_loss, test_loss)

    tf._set_optimzer(args.opt, lr_g=args.lr_g, lr_d = args.lr_d, weight_decay=args.weight_decay)
    tf.train(args.max_epoch, G_opt_step=1, D_opt_step=1, save_pre_epoch=1, print_pre_step=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--log_dir', type=str, required=True) 

    # data configuration
    parser.add_argument('--lmdb_train', default='../../../data/RealComData/realcom_data_train.lmdb')
    parser.add_argument('--lmdb_valid', default='../../../data/RealComData/realcom_data_test.lmdb')
    parser.add_argument('--lmdb_sn', default='../../../data/RealComShapeNetData/shapenet_data.lmdb')
    parser.add_argument('--class_name', default='chair', choices=['all', 'chair', 'table', 'trash_bin', 'tv_or_monitor', 'cabinet', 'bookshelf', 'sofa', 'lamp', 'bed', 'tub', 'car', 'chair, table', 'chair, trash_bin',\
                                                                   'table, sofa', 'bookshelf, tv_or_monitor', 'sofa, bed', 'tub, lamp', 'table, tv_or_monitor', 'trash_bin, chair'])
    parser.add_argument('--motivation', action='store_true', default=False)
    parser.add_argument('--balanced', action='store_true', default = False)
    parser.add_argument('--target_ratio', nargs='+', type = float, default=[1.0, 1.0])
    parser.add_argument('--pcn', action='store_true', default = False)

    # optimizer configuration
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr_d', type=float, default=1e-5)
    parser.add_argument('--lr_g', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--betas_g', nargs='+', type=float, default = [0.95, 0.999])
    parser.add_argument('--betas_d', nargs='+', type=float, default = [0.95, 0.999])
    parser.add_argument('--clip', type=float, default = 0)

    # train configuration
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_pn', type=int, default=2048)
    parser.add_argument('--gt_pn', type=int, default=2048)
    parser.add_argument('--max_epoch', type=int, default=480)
    parser.add_argument('--wo_comple', action='store_true', default=False)

    # discriminator loss
    parser.add_argument('--phi1', type=str ,default='softplus', choices=['linear', 'kl','sofplus'])
    parser.add_argument('--phi2', type=str, default='softplus', choices=['linear', 'kl','sofplus'])
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--dens_coord', default=10.5, type=float)
    parser.add_argument('--r1_gamma', default=0, type=float)
    parser.add_argument('--r2_gamma', default=0, type=float)

    # cost configuration
    parser.add_argument('--cost_type', default = 'bi', type=str,  choices=['uni', 'bi', 'mix'])
    parser.add_argument('--tau1', default = 100, type = float)
    parser.add_argument('--tau2', default = 100, type = float)
    parser.add_argument('--cost_function', default = 'infocd', type=str, choices=['cd', 'cd1', 'l2', 'infocd', 'emd', 'cd_fwd'])

    # additional loss configuration
    parser.add_argument('--TC', type=str, choices = ['T1', 'T2', 'T11', 'T12', 'T22', 'T12_OT'])
    parser.add_argument('--cut_grad', action='store_true', default=False)

    
    args = parser.parse_args()
    

    train(args)
