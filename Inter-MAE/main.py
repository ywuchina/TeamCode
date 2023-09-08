from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
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
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
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
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
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
    # print('args=>',args)
    # args = > Namespace(ckpts=None, config='cfgs/pretrain.yaml', deterministic=False, distributed=False,
    #                    exp_name='pointmae', experiment_path='./experiments/pretrain/cfgs/pointmae',
    #                    finetune_model=False, fold=-1, launcher='none', local_rank=0, log_name='pretrain', loss='cd1',
    #                    mode=None, num_workers=8, resume=False, scratch_model=False, seed=0, shot=-1, start_ckpts=None,
    #                    sync_bn=False, test=False, tfboard_path='./experiments/pretrain/cfgs/TFBoard/pointmae',
    #                    use_gpu=True, val_freq=1, vote=False, way=-1)
    # print('config=>',config)
    # config = > {'optimizer': {'type': 'AdamW', 'kwargs': {'lr': 0.001, 'weight_decay': 0.05}},
    #             'scheduler': {'type': 'CosLR', 'kwargs': {'epochs': 300, 'initial_epochs': 10}}, 'dataset': {'train': {
    #         '_base_': {'NAME': 'ShapeNet', 'DATA_PATH': '/home/ljm/data/ShapeNet55-34/ShapeNet-55', 'N_POINTS': 8192,
    #                    'PC_PATH': '/home/ljm/data/ShapeNet55-34/shapenet_pc'},
    #         'others': {'subset': 'train', 'npoints': 1024, 'bs': 32}}, 'val': {
    #         '_base_': {'NAME': 'ShapeNet', 'DATA_PATH': '/home/ljm/data/ShapeNet55-34/ShapeNet-55', 'N_POINTS': 8192,
    #                    'PC_PATH': '/home/ljm/data/ShapeNet55-34/shapenet_pc'},
    #         'others': {'subset': 'test', 'npoints': 1024, 'bs': 64}}, 'test': {
    #         '_base_': {'NAME': 'ShapeNet', 'DATA_PATH': '/home/ljm/data/ShapeNet55-34/ShapeNet-55', 'N_POINTS': 8192,
    #                    'PC_PATH': '/home/ljm/data/ShapeNet55-34/shapenet_pc'},
    #         'others': {'subset': 'test', 'npoints': 1024, 'bs': 32}}},
    #             'model': {'NAME': 'Point_MAE', 'group_size': 32, 'num_group': 64, 'loss': 'cdl2',
    #                       'transformer_config': {'mask_ratio': 0.6, 'mask_type': 'rand', 'trans_dim': 384,
    #                                              'encoder_dims': 384, 'depth': 12, 'drop_path_rate': 0.1,
    #                                              'num_heads': 6, 'decoder_depth': 4, 'decoder_num_heads': 6}},
    #             'npoints': 1024, 'total_bs': 32, 'step_per_update': 1, 'max_epoch': 300}
    # run
    if args.test:
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
