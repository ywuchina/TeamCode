import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData, ModelNet40
from model.SPCNet import SPCNet

from operations.transform_functions import PCRNetTransform
from losses.chamfer_loss import ChamferLoss
from losses.frobenius_norm import FrobeniusNormLoss
from losses.rmse_features import RMSEFeaturesLoss
from model.Autoencoder import Decoder_FC, Decoder_Hierarchical_FC, Decoder_Unconv, FI, PointNet
import  os


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('CFNet_Training')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--encoder', type=str, default='FI',
                        choices=['PointNet', 'FI'],
                        help='specify decoder type')
    parser.add_argument('--decoder', type=str, default='Decoder_FC',
                        choices=['Decoder_FC', 'Decoder_Unconv', 'Decoder_Hierarchical_FC'],
                        help='specify decoder type')
    parser.add_argument('--loss', type=str, default='',help='specify loss type')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--iter', type=int, default=2, help='number of iterations of SPCNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', type=int, default=200, help='epoch in training')
    parser.add_argument('--percent_train', type=float, default=1.0, help='only use part of training data')
    return parser.parse_args()

def npmat2euler(mats, seq='xyz'):
    eulers = []
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(mats)
    eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def find_errors(igt_R, pred_R, igt_t, pred_t):
    # igt_R:				Rotation matrix [3, 3] (source = igt_R * template)
    # pred_R: 			Registration algorithm's rotation matrix [3, 3] (template = pred_R * source)
    # igt_t:				translation vector [1, 3] (source = template + igt_t)
    # pred_t: 			Registration algorithm's translation matrix [1, 3] (template = source + pred_t)

    # Euler distance between ground truth translation and predicted translation.
    igt_t = -np.matmul(igt_R.T, igt_t.T).T			# gt translation vector (source -> template)
    translation_mse = np.sum(np.square(igt_t - pred_t))
    translation_mae = np.sum(np.abs(igt_t - pred_t))

    # Convert matrix remains to axis angle representation and report the angle as rotation error.
    igt_R_euler = npmat2euler(igt_R.T)
    pred_R_euler = npmat2euler(pred_R)
    rotation_mse = np.mean(np.square(igt_R_euler - pred_R_euler))
    rotation_mae = np.mean(np.abs(igt_R_euler - pred_R_euler))

    return rotation_mse, translation_mse, rotation_mae, translation_mae

def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
    errors_mse, errors_mae = [], []
    for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
        errors_mse.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i)[0:2])
        errors_mae.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i)[2:])
    return np.concatenate((np.sqrt(np.mean(errors_mse, axis=0)), np.mean(errors_mae, axis=0)))  # 一个batch的误差

def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt, igt_R, igt_t = data
        template = template.to(device)
        source = source.to(device)
        igt_R = igt_R.to(device)
        igt_t = igt_t.to(device)

        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source, args.iter)
        est_R = output['est_R']
        est_t = output['est_t']

        igt = PCRNetTransform.convert2transformation(igt_R, igt_t)  # [B,4,4]

        loss_gt = FrobeniusNormLoss()(output['est_T'], igt)  # 变换损失
        # print(output['est_T'].shape,igt.shape)
        loss_feature = RMSEFeaturesLoss()(output['feature_difference'])  # 特征损失
        loss_st = ChamferLoss()(template, output["transformed_source"])
        loss_re = ChamferLoss()(output['resource'], output['retemplate'])  # 重建损失
        loss_re2 = ChamferLoss()(template, output['retemplate']) + ChamferLoss()(source,output['resource'])  # 重建损失

        loss_val = loss_gt + loss_st + 0.001*loss_feature + 0.01*loss_re2

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss

def test_one_epoch(device, model, test_loader):
    model.eval()
    count = 0
    test_loss = 0
    errors=[]

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, igt_R, igt_t = data
        template = template.to(device)
        source = source.to(device)
        igt_R = igt_R.to(device)
        igt_t = igt_t.to(device)

        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source, args.iter)
        est_R = output['est_R']
        est_t = output['est_t']

        errors.append(compute_accuracy(igt_R.detach().cpu().numpy(), est_R.detach().cpu().numpy(),
                                       igt_t.detach().cpu().numpy(), est_t.detach().cpu().numpy()))

        igt = PCRNetTransform.convert2transformation(igt_R, igt_t)  # [B,4,4]

        loss_gt = FrobeniusNormLoss()(output['est_T'], igt)  # 变换损失
        # print(output['est_T'].shape,igt.shape)
        loss_feature = RMSEFeaturesLoss()(output['feature_difference'])  # 特征损失
        loss_st = ChamferLoss()(template, output["transformed_source"])
        loss_re = ChamferLoss()(output['resource'], output['retemplate'])  # 重建损失
        loss_re2 = ChamferLoss()(template, output['retemplate']) + ChamferLoss()(source, output['resource'])  # 重建损失

        loss_val = loss_gt + loss_st + 0.001 * loss_feature + 0.01 * loss_re2

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    errors = np.mean(np.array(errors), axis=0)
    return test_loss,errors[0],errors[1],errors[2],errors[3]


def train(model, train_loader, test_loader):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(learnable_params)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    # if checkpoint is not None:
    #     min_loss = checkpoint['min_loss']
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    wandb.init(project='CFNet', name=args.exp_name)
    wandb.watch(model)
    best_test_loss = np.inf
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(device, model, train_loader, optimizer)
        test_loss,er_rmse,et_rmse,er_mae,et_mae = test_one_epoch(device, model, test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/{}/models/{}/best_model_snap_{}b_7d_{}iter_{}.t7'.format(args.encoder,args.decoder,str(args.batch_size),str(args.iter),args.loss))
            torch.save(model.state_dict(), 'checkpoints/{}/models/{}/best_model_{}b_7d_{}iter_{}.t7'.format(args.encoder,args.decoder,str(args.batch_size),str(args.iter),args.loss))

        wandb_log = {}
        wandb_log['Train Loss'] = train_loss
        wandb_log['Test Loss'] = test_loss
        wandb_log['RMSE(R)'] = er_rmse
        wandb_log['MAE(R)'] = er_mae
        wandb_log['RMSE(t)'] = et_rmse
        wandb_log['MAE(t)'] = et_mae
        wandb.log(wandb_log)

        log_string("EPOCH:{},Training Loss:{},Testing Loss:{},Best Loss:{}".format(epoch + 1, train_loss, test_loss, best_test_loss))
        log_string(f"RMSE(Rotation Error: {er_rmse} & Translation Error: {et_rmse},MAE(Rotation Error: {er_mae} & Translation Error: {et_mae}")

if __name__ == '__main__':

    args = parse_args()
    # args.loss = 'loss_g+loss_p+loss_feature+loss_re2'   #与train_one_epoch和test_one_epoch上面语句保持一致

    START_EPOCH = 0
    MAX_EPOCHS = args.epoch

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    '''HYPER PARAMETER'''
    device = torch.device('cuda:{}'.format(args.gpu))

    '''LOG'''
    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger(args.encoder+'-'+args.decoder)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    exp_dir = Path('logs')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.encoder)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.decoder)
    exp_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler('{}/train_{}b_7d_{}iter_{}.txt'.format(exp_dir,str(args.batch_size),str(args.iter),args.loss))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log_string('os.getpid = %s'%os.getpid())
    log_string(args)

    trainset = RegistrationData(ModelNet40(train=True, gaussian_noise=args.noise, unseen=args.unseen, percent=args.percent_train))
    testset = RegistrationData(ModelNet40(train=False))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.batch_size//2, shuffle=False, drop_last=False, num_workers=4)

    Encoder = None
    if args.encoder == 'PointNet':
        Encoder = PointNet()
    elif args.encoder == 'FI':
        Encoder = FI()

    Decoder = None
    if args.decoder=='Decoder_FC':
        Decoder = Decoder_FC()
    elif args.decoder=='Decoder_Hierarchical_FC':
        Decoder = Decoder_Hierarchical_FC()
    elif args.decoder=='Decoder_Unconv':
        Decoder = Decoder_Unconv()

    model = SPCNet(Encoder,Decoder)
    model = model.to(device)

    resume = 'checkpoints/{}/models/{}/best_model_snap_{}b_7d_{}iter_{}.t7'.format(args.encoder,args.decoder,str(args.batch_size),str(args.iter),args.loss)  # 最新的检查点文件
    pretrained = 'checkpoints/{}/models/{}/best_model_{}b_7d_{}iter_{}.t7'.format(args.encoder,args.decoder,str(args.batch_size),str(args.iter),args.loss)  # 是否有训练过的模型可用

    checkpoint = None

    # if os.path.exists(resume):
    #     checkpoint = torch.load(resume)
    #     print(checkpoint.keys())
    #     START_EPOCH = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model'])
    # else:
    #     log_string('No pretrained model:{}'.format(resume))

    train(model, trainloader, testloader)
    log_string('End of training...\n')
