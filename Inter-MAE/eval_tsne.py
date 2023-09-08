import os
import torch
import torch.nn as nn
from tqdm import tqdm

from tools import builder
from utils import parser, dist_utils, misc
import time
from utils.logger import *
from utils.config import *
from utils.AverageMeter import AverageMeter
import matplotlib.pyplot as plt
import numpy as np
from datasets import data_transforms

if __name__ == "__main__":
    # args
    args = parser.get_args()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs

    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    base_model.cuda()

    feats_train = []
    labels_train = []
    base_model.load_model_from_ckpt(args.ckpts)
    base_model.eval()

    feats_test = []
    labels_test = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, 1024)

            logits, feats = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            feats_test.append(feats.detach())
            labels_test.append(label.detach())

        feats_test = torch.cat(feats_test, dim=0)
        labels_test = torch.cat(labels_test, dim=0)

    print('feats_test=>', feats_test.shape)
    print('labels_test=>', labels_test.shape)
    feats_test = np.array(feats_test.cpu())
    labels_test = np.array(labels_test.cpu())
    print('feats_test=>', feats_test.shape)
    print('labels_test=>', labels_test.shape)

    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca')
    X_tsne = tsne.fit_transform(feats_test)

    plt.figure(figsize=(8, 8))

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_test, cmap = 'cool')

    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('modelnet_joint')
    plt.savefig('modelnet_joint.png')






