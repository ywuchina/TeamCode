import argparse

import torch.multiprocessing
import wandb
from torch.optim.lr_scheduler import StepLR

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys, os, random
sys.path.append("../..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from others.pointnetlk.data_utils import RegistrationkittiData, Kitti, ModelNet40
from PointNetLK_model import PointLK, PointNet_features
from evaluate_funcs import calculate_R_msemae, calculate_t_msemae, compute_batch_error
from visualization import CloudVisualizer

# use_cuda = torch.cuda.is_available()
# torch.cuda.set_device(0)

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(1234)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNetLK_Training')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', type=int, default=100, help='epoch in training')
    parser.add_argument('--percent_train', type=float, default=1.0, help='only use part of training data')
    return parser.parse_args()

def train_one_epoch(net, train_loader, optimizer):
    net.train()
    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    for i, data in enumerate(tqdm(train_loader)):
        src, target, rotation, translation, euler = data
        # print(src.shape, target.shape)
        batchsize = src.shape[0]
        src = src.to(device)
        target = target.to(device)
        rotation = rotation.to(device)
        translation = translation.to(device)

        # translation = translation - torch.mean(translation, dim=1, keepdim=True).to(translation)
        # src = src - torch.mean(src, dim=2, keepdim=True).to(src)
        # target = target - torch.mean(target, dim=2, keepdim=True).to(target)

        rotation_pred, translation_pred, r = net(src, target)
        num_examples += batchsize

        # 保存rotation and translation, detach()后不再计算梯度
        rotations.append(rotation.detach().cpu().numpy())
        translations.append(translation.detach().cpu().numpy())
        rotations_pred.append(rotation_pred.detach().cpu().numpy())
        translations_pred.append(translation_pred.detach().cpu().numpy())

        # 变换点云
        transformed_src = torch.bmm(rotation_pred, src) + translation_pred.reshape(batchsize, 3, 1)
        # print('src.shape=',src.shape,'\ntarget.shape=',target.shape)
        # vis = CloudVisualizer(0.01, os.path.join('../result','pointnetlk', "{}_pointnetlk_kitti".format(str(i))))
        # vis.reset(src.permute(0,2,1)[1, :, :3].cpu().numpy(),target.permute(0,2,1)[1, :, :3].cpu().numpy(),
        #           transformed_src.permute(0,2,1)[1, :, :3].detach().cpu().numpy())

        # 再三个维度上分别复制batch_size次, 1次, 1次，单位矩阵E用于计算损失
        E = torch.eye(3).unsqueeze(0).repeat(batchsize, 1, 1).to(device)
        loss_T = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), E) \
                 + F.mse_loss(translation_pred, translation)

        z = torch.zeros_like(r)
        loss_r = torch.nn.functional.mse_loss(r, z, size_average=False)
        loss = loss_T + loss_r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batchsize

    return total_loss * 1.0 / num_examples

def test_one_epoch(net, test_loader):
    net.eval()
    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            src, target, rotation, translation, euler = data
            # print(src.shape, target.shape)
            batchsize = src.shape[0]
            src = src.to(device)
            target = target.to(device)
            rotation = rotation.to(device)
            translation = translation.to(device)

            # translation = translation - torch.mean(translation, dim=1, keepdim=True).to(translation)
            # src = src - torch.mean(src, dim=2, keepdim=True).to(src)
            # target = target - torch.mean(target, dim=2, keepdim=True).to(target)

            rotation_pred, translation_pred, r = net(src, target)

            num_examples += batchsize

            # 保存rotation and translation, detach()后不再计算梯度
            rotations.append(rotation.detach().cpu().numpy())
            translations.append(translation.detach().cpu().numpy())
            rotations_pred.append(rotation_pred.detach().cpu().numpy())
            translations_pred.append(translation_pred.detach().cpu().numpy())

            # 变换点云
            transformed_src = torch.bmm(rotation_pred, src) + translation_pred.reshape(batchsize, 3, 1)
            # print('src.shape=',src.shape,'\ntarget.shape=',target.shape)
            # vis = CloudVisualizer(0.01, os.path.join('../result','pointnetlk', "{}_pointnetlk_kitti".format(str(i))))
            # vis.reset(src.permute(0,2,1)[1, :, :3].cpu().numpy(),target.permute(0,2,1)[1, :, :3].cpu().numpy(),
            #           transformed_src.permute(0,2,1)[1, :, :3].detach().cpu().numpy())

            # 再三个维度上分别复制batch_size次, 1次, 1次，单位矩阵E用于计算损失
            E = torch.eye(3).unsqueeze(0).repeat(batchsize, 1, 1).to(device)
            loss_T = F.mse_loss(torch.matmul(rotation_pred.transpose(2, 1), rotation), E) \
                     + F.mse_loss(translation_pred, translation)

            z = torch.zeros_like(r)
            loss_r = torch.nn.functional.mse_loss(r, z, size_average=False)

            loss = loss_T + loss_r

            total_loss += loss.item() * batchsize

        rotations = np.concatenate(rotations, axis=0)
        translations = np.concatenate(translations, axis=0)
        rotations_pred = np.concatenate(rotations_pred, axis=0)
        translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations, \
           translations, rotations_pred, translations_pred

def train(model, train_loader, test_loader):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(learnable_params)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    # if checkpoint is not None:
    #     min_loss = checkpoint['min_loss']
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    wandb.init(project='PointNetLK', name=args.exp_name)
    wandb.watch(model)
    best_test_loss = np.inf
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        test_loss, test_R, test_t, test_R_pred, test_t_pred = test_one_epoch(model, test_loader)

        test_errors = compute_batch_error(test_R, test_R_pred, test_t, test_t_pred)
        test_R_mse, test_R_mae = calculate_R_msemae(test_R, test_R_pred)
        test_R_rmse = np.sqrt(test_R_mse)
        test_t_mse, test_t_mae = calculate_t_msemae(test_t, test_t_pred)
        test_t_rmse = np.sqrt(test_t_mse)

        R_rmse.append(test_R_rmse)
        R_mae.append(test_R_mae)
        t_rmse.append(test_t_rmse)
        t_mae.append(test_t_mae)
        print('Test: Loss: %f, MSE(R): %f, RMSE(R): %f, MAE(R): %f, '
              'MSE(t): %f, RMSE(t): %f, MAE(t): %f, error(t): %f, '
              'error(R): %f, error(matR): %f'
              % (test_loss, test_R_mse, test_R_rmse, test_R_mae, test_t_mse,
                 test_t_rmse, test_t_mae, test_errors[0], test_errors[1], test_errors[2]))

        wandb_log = {}
        wandb_log['Train Loss'] = train_loss
        wandb_log['Test Loss'] = test_loss
        wandb_log['RMSE(R)'] = test_R_rmse
        wandb_log['MAE(R)'] = test_R_mae
        wandb_log['RMSE(t)'] = test_t_rmse
        wandb_log['MAE(t)'] = test_t_mae
        wandb.log(wandb_log)

if __name__ == '__main__':
    args = parse_args()

    print(args)
    START_EPOCH = 0
    MAX_EPOCHS = args.epoch
    max_angle = 45
    max_t = 0.5
    # partial_source = False
    # unseen = False
    # noise = False
    single = -1
    R_rmse, R_mae, t_rmse, t_mae = [], [], [], []
    train_loader = DataLoader(
        ModelNet40(2048, partition='train', max_angle=max_angle, max_t=max_t, noise=args.noise,unseen = args.unseen,
                              single = -1, percent = args.percent_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(
        ModelNet40(2048, partition='test', max_angle=max_angle, max_t=max_t, single=-1,percent=1.0),
        batch_size=args.batch_size//2, shuffle=False, drop_last=False, num_workers=4)

    # testset = Registration3dmatchData(ThreeDMatch(split='val'), sample_point_num=2048, max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=numworkers)

    # trainset = RegistrationkittiData(Kitti(split='train', type='tracking'), sample_point_num=2048, max_angle=max_angle,
    #                                 max_t=max_t, unseen=unseen, noise=noise, single=single, is_testing=True)
    # print('train',len(trainset))
    # testset= RegistrationkittiData(Kitti(split='test',type='tracking'), sample_point_num=2048 ,max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # print('test', len(testset))

    # print(len(test_loader))
    device = torch.device('cuda:{}'.format(args.gpu))
    ptnet = PointNet_features(emb_dims=1024, symfn='max')

    Rnet = PointLK(ptnet=ptnet)
    # checkpoint = torch.load('./pointnetlk.pth', map_location=lambda storage, loc: storage.cuda(0))
    # Rnet.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # Rnet = Rnet.cuda()
    # Rnet.eval()
    Rnet = Rnet.to(device)
    train(Rnet, train_loader, test_loader)

