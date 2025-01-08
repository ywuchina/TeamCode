import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder import Encoder
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic

class MINet(nn.Module):
    def __init__(self,feature_model=Encoder()):
        super(MINet, self).__init__()
        self.feature_model=feature_model
        self.fc1 = nn.Linear(1024 * 2, 1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2=nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3=nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4=nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5=nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 8)


    def forward(self, template, source, max_iteration=3):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        template_features, template_encoder_loss=self.feature_model(template)

        for i in range(max_iteration):
            est_R, est_t, source, self.source_encoder_loss, self.source_features= \
                self.spam(template_features, source, est_R, est_t)
        result = {
                  'template_features':template_features,
                  'source_features':self.source_features,
                  'template_encoder_loss':template_encoder_loss,
                  'source_encoder_loss':self.source_encoder_loss,
                  'transformed_source': source,     
                  'est_R': est_R,  
                  'est_t': est_t,  
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                 }
        return result

    def spam(self, template_features, source, est_R, est_t):
        source_features, source_encoder_loss = self.feature_model(source)
        batch_size = source.size(0)
        y = torch.cat([template_features, source_features], dim=1)

        pose_8d = F.relu(self.fc1(y))
        pose_8d = F.relu(self.fc2(pose_8d))
        pose_8d = F.relu(self.fc3(pose_8d))
        pose_8d = F.relu(self.fc4(pose_8d))
        pose_8d = F.relu(self.fc5(pose_8d))  # [B,8]
        pose_8d=self.fc6(pose_8d)

        pose_8d = PCRNetTransform.create_pose_8d(pose_8d)

        real_part = pose_8d[:, 0:4]
        dual_part = pose_8d[:, 4:]
        est_R_temp, est_t_temp = dual_quat_to_extrinsic(real_part, dual_part)
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        est_R = torch.bmm(est_R_temp, est_R)
        source = PCRNetTransform.quaternion_transform2(source, pose_8d, est_t_temp)

        return est_R, est_t, source, source_encoder_loss, source_features

