import os

import numpy as np
import torch

from model.SPCNet import SPCNet
from model.Autoencoder import Decoder_Unconv, FI,Decoder_FC,Decoder_Hierarchical_FC
MODEL = SPCNet(FI(),Decoder_FC())

from thop import profile
dummy_input1 = torch.randn(1, 1024, 3)
dummy_input2 = torch.randn(1, 1024 ,3)
flops, params = profile(MODEL, (dummy_input1,dummy_input2,))
print('flops: ', flops, 'params: ', params)
print('flops: %.4f G, params: %.4f M' % (flops / 1000000000.0, params / 1000000.0))
assert 1 == 2

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import RegistrationData,ModelNet40
from losses.chamfer_loss import ChamferLoss
# import transforms3d
from losses.rmse_features import RMSEFeaturesLoss
from losses.frobenius_norm import FrobeniusNormLoss
from operations.transform_functions import PCRNetTransform
from model.SPCNet import SPCNet
from model.Autoencoder import Decoder_Unconv, FI,Decoder_FC,Decoder_Hierarchical_FC
from operations.visualization import visual_pcd
from operations.tools import IOStream
from scipy.spatial.transform import Rotation

from visualization import CloudVisualizer

BATCH_SIZE=32
NUM_WORKERS = 4
EVAL=False
START_EPOCH=0
MAX_EPOCHS=200

exp_name = 'FI'
pretrained='checkpoints/best_model.t7'.format(exp_name)       #使用最好的模型参数测试
device =torch.device("cuda:0")
MODEL = SPCNet(FI(),Decoder_FC())

textio = IOStream('checkpoints/' + exp_name+"/" + 'run_test.log')
textio = IOStream('checkpoints/{}/run_test_un.log'.format(exp_name))


def npmat2euler(mats, seq='xyz'):
    eulers = []
    
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
    t_error_mse =  np.sum(np.square(igt_t - pred_t))
    t_error_mae = np.sum(np.abs(igt_t - pred_t))

    error_mat = np.dot(igt_R, pred_R)
    # _, angle = transforms3d.axangles.mat2axangle(error_mat)
    angle = 0

    igt_R_euler = npmat2euler(igt_R.T)
    pred_R_euler = npmat2euler(pred_R)
    r_error_mse = np.mean((igt_R_euler - pred_R_euler) ** 2)
    r_error_mae = np.mean(np.abs(igt_R_euler - pred_R_euler))

    return abs(angle*(180/np.pi)), r_error_mse, t_error_mse, r_error_mae, t_error_mae

def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
    errors_mse,errors_mae = [],[]
    for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
        errors_mse.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i)[0:3])
        errors_mae.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i)[3:])
    return np.concatenate((np.sqrt(np.mean(errors_mse, axis=0)),np.mean(errors_mae, axis=0)))  #一个batch的误差

def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    errors = []

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, igt_R, igt_t = data
        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        igt_t = igt_t - torch.mean(igt_t, dim=1).unsqueeze(1).to(igt_t)
        source = source - torch.mean(source, dim=1, keepdim=True).to(source)
        template = template - torch.mean(template, dim=1, keepdim=True).to(template)
        output = model(template, source)
        est_R = output['est_R']
        est_t = output['est_t']
        errors.append(compute_accuracy(igt_R.detach().cpu().numpy(), est_R.detach().cpu().numpy(),
                                       igt_t.detach().cpu().numpy(), est_t.detach().cpu().numpy()))

        transformed_source = torch.bmm(est_R, source.permute(0, 2, 1)).permute(0, 2, 1) + est_t
        # display_open3d(template.detach().cpu().numpy()[0], source_original.detach().cpu().numpy()[0], transformed_source.detach().cpu().numpy()[0])
        vis = CloudVisualizer(0.01, os.path.join('result', 'spcnet', "{}_spcnet".format(str(i))))
        vis.reset(source[1, :, :3].detach().cpu().numpy(), template[1, :, :3].detach().cpu().numpy(),
                  output["transformed_source"][1, :, :3].detach().cpu().numpy())


        # 7/8d 转变换矩阵
        igt = igt.squeeze(1).contiguous()
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(source.size(0), 3, 3).contiguous()
        est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
        est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
        igt = PCRNetTransform.convert2transformation(est_R, est_t)  # [B,4,4]

        loss_g = FrobeniusNormLoss()(output['est_T'], igt)
        loss_p = ChamferLoss()(template,output["transformed_source"])
        loss_feature = RMSEFeaturesLoss()(output['feature_difference'])
        loss_re = ChamferLoss()(template, output['retemplate']) + ChamferLoss()(source, output['resource'])
        loss_val = loss_g+loss_re+loss_p+0.01*loss_feature
        visual_pcd(template.detach().cpu().numpy()[3],source.detach().cpu().numpy()[3],output["transformed_source"].detach().cpu().numpy()[3])
        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss)/count
    errors = np.mean(np.array(errors), axis=0)
    return test_loss, errors[0], errors[1],errors[2]

if __name__ == '__main__':

    testset = RegistrationData(ModelNet40(train=False))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    model = MODEL
    model = model.to(device)
    if pretrained:
        model.load_state_dict(torch.load(pretrained,map_location=device))
    total_test_loss = 0
    total_translation_error = 0
    total_rotation_error=  0
    total_rotation_error_xyz = 0
    test_loss, translation_error, rotation_error,rotation_error_xyz = test_one_epoch(device, model, testloader)
    total_test_loss = total_test_loss+test_loss
    total_translation_error = total_translation_error+translation_error
    total_rotation_error = total_rotation_error+rotation_error
    total_rotation_error_xyz = total_rotation_error_xyz+rotation_error_xyz

    print("Test Loss: {}, Rotation_pcr_rmse_Error: {} & Rotation_xyz_rmse_Error {} & Translation Error: {} ".format(test_loss, rotation_error,rotation_error_xyz,translation_error))
    textio.cprint("Test Loss: {}, Rotation_pcr_rmse_Error: {} & Rotation_xyz_rmse_Error {} & Translation Error: {} ".format(test_loss, rotation_error,rotation_error_xyz,translation_error))
    textio.cprint("-----------------------------------------------------")