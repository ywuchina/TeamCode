import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False,
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test for certain ckpt')
    parser.add_argument(
        '--test_svm', 
        choices=['modelnet40', 'scan'],
        default=None,
        help = 'test_svm for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode

    # args.data_root = '/home/wzp/DATA/SHREC2020-CD'
    # args.data_root = '/home/wzp/DATA/SHREC2020-CD/no_plane_data'
    # args.data_root = '/home/wzp/DATA/SHREC2020-CD/rgb_norm_data/'
    args.data_root = '/home/wzp/DATA/SHREC2020-CD/norm_data/'
    args.datapath = os.path.join(args.data_root, 'train_seg') 
    args.n_samples = 8192
    args.if_prepare_data = True # saving processed PCs to .npy to accelerate training phase. 
    args.prepare_data = args.data_root + '/prapared_data_' + str(args.n_samples) 
    args.txtpath = os.path.join(args.data_root, './data/train.txt') 
    args.test_datapath = os.path.join(args.data_root, 'test_seg') 
    args.test_txtpath = os.path.join(args.data_root, './data/test.txt') 

   
    args.val_txtpath = os.path.join(args.data_root, './data/val.txt') 
    args.config = './cfgs/pre-training/point-m2ae.yaml'
    args.kn = 8
    args.exp_dir = '/home/wzp/projects/Point-M2AE-main/outputs/SiamKPConv-2023-11-15/SiamKPConv-2023-11-15-SiameseKPConv-20231115_231513/'
    
    args.resume_path = os.path.join(args.exp_dir, 'experiments', Path(args.config).parent.stem, Path(args.config).stem, args.exp_name)

    # args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem, Path(args.config).stem,
    #                                     'no_fpc',args.exp_name+'k' + str(args.kn))
    
    # args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem, Path(args.config).stem,
    #                                     'Lrec')
    
    # args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem, Path(args.config).stem,'rgb_save',
    #                                     args.exp_name+'k' + str(args.kn))
    args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem, Path(args.config).stem,
                                        args.exp_name+'k' + str(args.kn))
    # args.experiment_path = os.path.join('./experiments', Path(args.config).parent.stem, Path(args.config).stem,
    #                                     args.exp_name)
    args.save_txt_path = os.path.join(args.experiment_path,'f_TEXT')
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    
