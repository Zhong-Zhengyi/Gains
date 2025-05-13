import argparse
from models.digit5 import *
from models.amazon import *
from dataset import data_process
import yaml
from algs.fl_base import Base
from algs.ours import Ours
import torch
import numpy as np
import copy


def get_args():
    parser = argparse.ArgumentParser(description='federated adaption setting')
    parser.add_argument('--base_path', default="./")
    parser.add_argument('--paradigm', default='motivation', type=str, help='choose the training paradigm:  pretrain, ours, fedavg, fedprox, fedproto, fosda, semifda, autofedgp, fedheal')
    parser.add_argument('--shift', default='medium', type=str, help='mild, medium, strong')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--num_user', default=50, type=int, help='number of clients')
    parser.add_argument('--frac', default=1, type=float, help='the faction of clients participating in the training')
    parser.add_argument('--data', type=str, default='digitfive', help='digitfive, amazonreview')
    parser.add_argument('--global_epoch', type=int, default=2)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_parallel', type=bool, default=False)
    parser.add_argument('--target_domain', default=[1,5], help='mnistm, mnist, syn, usps, svhn, books, dvd, electronics, kitchen_housewares,[1,5]')
    parser.add_argument('--source_domain', default='mnist', help=['mnistm', 'mnist', 'syn', 'usps', 'svhn', 'books', 'dvd', 'electronics', 'kitchen'])
    parser.add_argument('--alpha', type=float, default=0.5)
    #===============fosda===============
    parser.add_argument('--beta', type=float, default=0.5)

    ### mild #####
    parser.add_argument('--mild_domain', default='mnist', help=['mnistm', 'mnist', 'syn', 'usps', 'svhn'])
    args = parser.parse_args()
    return args

args = get_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    if args.data == 'digitfive': 
        args.img_sz = 32
        args.in_channel = 3
        args.num_classes = 10
        model = CNNFullModel(args.data_parallel).to(args.device)
        if args.shift == 'strong':
            args.num_user = 5
    elif args.data == 'amazonreview':
        args.img_sz = 300
        args.in_channel = 3
        args.num_classes = 2
        model = AmazonFullLSTM(args.data_parallel).to(args.device)
        if args.shift == 'strong':
            args.num_user = 4
    args.new_users = [args.num_user-1]
    train_loaders, test_loaders, args = data_process.data_distri(args)
    
    # pre-training on source domain
    if args.paradigm == 'pretrain':
        trainer = Base(args)
        pre_model = trainer.FL_Train(model, train_loaders[:-1], test_loaders[:-1], args)

    # knowledge adaptation
    if args.paradigm == 'ours':
        # load source global model
        pre_param = torch.load('./save_model/pretrained_globalmodel_shift_{}_{}+{}.pth'.format(args.shift, args.target_domain, args.source_domain))
        model.load_state_dict(pre_param)
        pre_model = model
        trainer = Ours(args)
        model= trainer.FL_Train(pre_model, train_loaders, test_loaders, args)