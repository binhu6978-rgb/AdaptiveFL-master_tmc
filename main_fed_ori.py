#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
from tqdm import tqdm

from Algorithm.Training_Decoupled import Decoupled
from Algorithm.Training_ScaleFL import ScaleFL
# NOTE: models.transformer imports transformers+datasets; it is only needed
# by the 'transformer' model branch, so it is imported lazily there.

# matplotlib.use('Agg')
import copy

from utils.Clients import Clients
from utils.options import args_parser
from models import *
from utils.get_datatset_new import get_dataset
from utils.utils import save_result
from utils.set_seed import set_random_seed
from Algorithm.Training_AdaptiveFL import AdaptiveFL
from Algorithm.Training_HeteroFL import HeteroFL
from Algorithm.Training_SemConsFL import SemConsFL

import sys
import logging
from datetime import datetime
# =============================================================================
# Tee logger: print to terminal + file
# =============================================================================
class TeeLogger:
    """Write stdout to both console and a file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def f2s(x: float) -> str:
    # 0.6 -> "0p6"  (避免文件名里出现多个点或奇怪格式)
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    print(net_glob)

    # training
    acc = []
    clients = Clients(args)
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"this epoch choose: {idxs_users}")
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

            clients.train(idx, iter)
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(round=iter, net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)
# 定义一个类用于重定向标准输出到日志文件
class RedirectStdoutToLog:
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if message.strip():  # 避免记录空消息
            self.logger.info(message.strip())

    def flush(self):
        pass
# 重定向标准输出到日志
sys.stdout = RedirectStdoutToLog(logging.getLogger())
import os
if __name__ == '__main__':
    # parse args
    run_dir = 'result_ori_resnet_finsih'
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # redirect stdout to log file (tee)


    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Keep existing --algorithm usable; this entry point now defaults to SemConsFL.
    explicit_algorithm = any(x == '--algorithm' or x.startswith('--algorithm=') for x in sys.argv[1:])
    algorithms = [args.algorithm if explicit_algorithm else 'SemConsFL']

    for algorithm in algorithms:
        args.data_beta = 0.3
        # args.epochs=5
        args.model = 'resnet'
        args.dataset = 'cifar10'
        args.algorithm = algorithm
        if algorithm == 'SemConsFL':
            args.iid = 0
            args.run_dir = os.path.join('result_semconsfl', stamp)
            os.makedirs(args.run_dir, exist_ok=True)

        if args.dataset.lower() == 'cifar10':
            args.num_classes = 10
        elif args.dataset.lower() == 'cifar100':
            args.num_classes = 100
        elif args.dataset.lower() in ['tinyimagenet', 'tiny_imagenet', 'tiny-imagenet']:
            args.num_classes = 200
        else:
            args.num_classes = 10  # 默认值，可根据需要修改



        log_path = os.path.join(
            args.run_dir if algorithm == 'SemConsFL' else run_dir,
            "%s_%s_%s_%s_%s.log" % (str(args.data_beta), args.dataset, args.algorithm, args.model, stamp))


        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_path,
                            filemode='a')


        set_random_seed(args.seed)

        dataset_train, dataset_test, dict_users = get_dataset(args)
        # plot_client_distribution(dataset_train, dict_users)

        if args.model == 'cnn':
            if args.dataset == 'femnist':
                net_glob = CNNFashionMnist(args)
            elif args.dataset == 'mnist':
                net_glob = CNNMnist(args)
            elif args.use_project_head:
                net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
            elif 'cifar' in args.dataset:
                net_glob = CNNCifar(args)
        elif 'resnet' in args.model:
            net_glob = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)  # 默认为3通道
        elif 'mobilenet' in args.model:
            net_glob = MobileNetV2(args.num_channels, args.num_classes)
        elif 'vgg' in args.model:
            net_glob = vgg_16_bn(num_classes=args.num_classes, track_running_stats=True, num_channels=args.num_channels)
        elif 'lstm' in args.model:
            net_glob = CharLSTM()
        elif 'transformer' in args.model:
            from models.transformer import Transformer
            net_glob = Transformer(vocab_size=30522, d_model=128, nhead=8, num_encoder_layers=8, max_len=256,
                                slim_idx=2,
                                scale=0.5, dropout=0.1)
        if args.algorithm == 'FedAvg':
            net_glob.to(args.device)
            FedAvg(net_glob, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'FedProx':
            FedProx(net_glob, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'AdaptiveFL':
            args.depth_saved = [2, 3, 4]
            args.width_ration = [0.4, 0.66, 1.0]
            AdaptiveFL(args, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'SemConsFL':
            args.depth_saved = [2, 3, 4]
            args.width_ration = [0.4, 0.66, 1.0]
            SemConsFL(args, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'HeteroFL':
            args.depth_saved = [0]
            args.width_ration = [0.5, 0.71, 1.0]
            HeteroFL(args, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'ScaleFL':
            args.depth_saved = [2, 3, 4]
            args.width_ration = [0.4, 0.66, 1.0]
            ScaleFL(args, dataset_train, dataset_test, dict_users)
        elif args.algorithm == 'Decoupled':
            args.depth_saved = [8]
            args.width_ration = [0.4, 0.66, 1.0]
            Decoupled(args, dataset_train, dataset_test, dict_users)
        else:
            raise "%s algorithm has not achieved".format(args.algorithm)
