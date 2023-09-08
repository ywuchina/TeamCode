import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np

from .ResNet import ResNet
from .ViT import ViT
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import emd
from lightly.loss import NTXentLoss
from torchvision.models import resnet50, resnet34, resnet101
from timm.models.vision_transformer import vit_base_patch16_224, vit_small_patch16_224, vit_base_patch16_384
from timm.models.vision_transformer import Attention, Mlp, Block

class TransPooling(nn.Module):
    def __init__(self, channels):
        super(TransPooling, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pts_feat):
        # B*N, C, K
        x_q = self.q_conv(pts_feat).permute(0, 2, 1)
        x_k = self.k_conv(pts_feat)
        attn = self.softmax(x_q @ x_k)
        attn = attn / (1e-8 + attn.sum(dim=-1, keepdim=True))
        x_attn = torch.einsum('bnn,bnc->bnc', attn, pts_feat.permute(0, 2, 1)).permute(0, 2, 1)
        return x_attn

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False)
        )
        self.first_trans = TransPooling(256)
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1, bias=False)
        )
        #self.second_trans = TransPooling(self.encoder_channel)

    def forward(self, point_groups):
        '''
            point_groups : B G K 3
            -----------------
            feature_global : B G C
        '''
        bs, g, k , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, k, 3)
        #point_groups = point_groups.permute(0, 3, 1, 2)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # bs*g 256 k
        #feature = torch.max(feature,dim=-1,keepdim=False)[0]  # bs*g 256 k
        feature_first = self.first_trans(feature) # bs*g 256 k
        feature_first_mean = feature_first.mean(dim=-1, keepdim=False)  # bs*g 256
        feature_first_max = feature_first.max(dim=-1,keepdim=False)[0] # bs*g 256
        # feature_first = feature_first_max.reshape(bs, g, 256)
        feature_first = torch.cat([feature_first_mean.reshape(bs, g, 256),feature_first_max.reshape(bs, g, 256)],dim = -1)
        #feature = torch.cat([feature_first_mean.expand(-1,-1,g), feature_first], dim=1)# BG 512 k

        feature_second = self.second_conv(feature_first.transpose(2,1)) # bs self.encoder_channel g
        #feature_second = self.second_trans(feature_second) # bs self.encoder_channel g
        return feature_second.permute(0, 2, 1)

# class Encoder(nn.Module):   ## Embedding module
#     def __init__(self, encoder_channel):
#         super().__init__()
#         self.encoder_channel = encoder_channel
#         self.first_conv = nn.Sequential(
#             nn.Conv2d(3, 128, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 1, bias=False)
#         )
#         self.first_trans = TransPooling(256)
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1, bias=False),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, self.encoder_channel, 1, bias=False)
#         )
#         self.second_trans = TransPooling(self.encoder_channel)
#
#     def forward(self, point_groups):
#         '''
#             point_groups : B G K 3
#             -----------------
#             feature_global : B G C
#         '''
#         bs, g, k , _ = point_groups.shape
#         point_groups = point_groups.permute(0, 3, 1, 2)
#         # encoder
#         feature = self.first_conv(point_groups)  # B 256 g k
#         feature = torch.max(feature,dim=-1,keepdim=False)[0]  # BG 256 g
#         feature_first = self.first_trans(feature) # BG 256 g
#         feature_first_mean = feature_first.mean(dim=-1,keepdim=True) # BG 256 1
#         feature = torch.cat([feature_first_mean.expand(-1,-1,g), feature_first], dim=1)# BG 512 g
#         feature = self.second_conv(feature) # BG self.encoder_channel g
#         # feature = torch.max(feature, dim=-1, keepdim=False)[0] # B self.encoder_channel g
#         feature_second = self.second_trans(feature) # B self.encoder_channel g
#         return feature_second.permute(0, 2, 1)

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x))  # only return the mask tokens predict pixel
        return x, x[:, -return_token_num:]


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.embed_dim = config.transformer_config.embed_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.embed_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Inter_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Inter_MAE] ', logger ='Inter_MAE')
        self.config = config
        self.embed_dim = config.transformer_config.embed_dim
        self.decoder_embed_dim = config.transformer_config.decoder_embed_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.decoder_embed_dim)
        )

        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Inter_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Inter_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # projection head
        self.pro_head = nn.Sequential(
            nn.Linear(self.decoder_embed_dim * 2, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias = False)
        )
        # prediction head
        self.pre_head = nn.Sequential(
            nn.Conv1d(self.decoder_embed_dim, 3*self.group_size, 1)
        )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_low = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_low = ChamferDistanceL2().cuda()
        else:
            self.loss_low = emd().cuda()
            # self.loss_func = emd().cuda()

    def forward(self, pts, imgs = None, vis = False, **kwargs):
        # print('pts=>', pts.shape, 'imgs=>', imgs.shape)
        # pts=> torch.Size([32, 1024, 3]) imgs=> torch.Size([32, 3, 224, 224])
        neighborhood, center = self.group_divider(pts)
        # print('neighborhood=>', neighborhood.shape, 'center=>', center.shape)
        # neighborhood = > torch.Size([32, 64, 32, 3])
        # center = > torch.Size([32, 64, 3])
        pc_vis, mask = self.MAE_encoder(neighborhood, center)
        # print('pc_vis=>', pc_vis.shape, 'mask=>', mask.shape, mask)
        # x_vis = > torch.Size([32, 26, 384])
        # mask = > torch.Size([32, 64])
        # tensor([[ True, False, False,  ..., False,  True, False],
        # [False,  True, False,  ..., False, False, False],
        # [ True,  True,  True,  ...,  True,  True, False],
        # ...,
        # [ True,  True,  True,  ..., False,  True,  True],
        # [False, False,  True,  ...,  True,  True,  True],
        # [ True,  True,  True,  ...,  True, False,  True]], device='cuda:0')
        B, V, C = pc_vis.shape

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, self.decoder_embed_dim)
        # print('pos_emd_vis=>', pos_emd_vis.shape) pos_emd_vis=> torch.Size([32, 26, 256])
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, self.decoder_embed_dim)
        # print('pos_emd_mask=>', pos_emd_mask.shape) pos_emd_mask=> torch.Size([32, 38, 256])

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        pc_full = torch.cat([pc_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        # print('pc_full=>', pc_full.shape, 'pos_full=>', pos_full.shape)
        # pc_full = > torch.Size([32, 64, 384])
        # pos_full = > torch.Size([32, 64, 256])
        pc_full = self.decoder_embed(pc_full)
        pc, pc_rec = self.MAE_decoder(pc_full, pos_full, N)
        B, VM, C = pc.shape
        # pc1 = F.adaptive_max_pool1d(pc.transpose(1, 2), 1).view(B, -1)
        # pc2 = F.adaptive_avg_pool1d(pc.transpose(1, 2), 1).view(B, -1)
        # pc = torch.cat((pc1, pc2), 1) # B C+C
        # pc = pc[:, 0]
        pc = torch.cat([pc[:, 0], pc[:, 1:].max(1)[0]], dim=-1)
        pc_fea = self.pro_head(pc) # B 256

        # image feature
        loss_fea = 0
        if imgs != None:
            loss_high = NTXentLoss(temperature = 0.1).cuda()
            img_head = ResNet(resnet50(), feat_dim=2048)
            img_head = ViT(vit_small_patch16_224(), feat_dim= 768).cuda()

            img_feats = img_head(imgs[0])
            for i in range(1,len(imgs)):
                img_feat = img_head(imgs[i])
                img_feats += img_feat
            img_feats_mean = img_feats / len(imgs)
            # img_feats_cat = torch.cat(img_feats)

            # img_fea = img_head(imgs)  # B 256
            # print('pc_fea=>', pc_fea.shape, 'img_feats_mean=>', img_feats_mean.shape)
            loss_fea = 0.0001 * loss_high(pc_fea, img_feats_mean)

        # point cloud reconstruction
        B, M, C = pc_rec.shape
        rebuild_points = self.pre_head(pc_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B*M S 3
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        # print('rebuild_points=>', rebuild_points.shape, 'gt_points=>', gt_points.shape)
        # rebuild_points = > torch.Size([1216, 32, 3])
        # gt_points = > torch.Size([1216, 32, 3])
        loss_pc = self.loss_low(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss_pc, loss_fea

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        print('neighborhood=>', neighborhood.shape)
        # neighborhood=> torch.Size([32, 64, 32, 3])
        print('center=>', center.shape)
        # center=> torch.Size([32, 64, 3])
        group_input_tokens = self.encoder(neighborhood)  # B G N
        print('group_input_tokens=>', group_input_tokens.shape)
        # group_input_tokens=> torch.Size([32, 64, 384])

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        print('cls_tokens=>', cls_tokens.shape)
        # cls_tokens = > torch.Size([32, 1, 384])
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        print('cls_pos=>', cls_pos.shape)
        # cls_pos = > torch.Size([32, 1, 384])

        pos = self.pos_embed(center)
        print('pos=>', pos.shape)
        # pos = > torch.Size([32, 64, 384])

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        print('x=>', x.shape)
        # x = > torch.Size([32, 65, 384])
        pos = torch.cat((cls_pos, pos), dim=1)
        print('pos=>', pos.shape)
        # pos = > torch.Size([32, 65, 384])
        # transformer
        x = self.blocks(x, pos)
        print('x=>', x.shape)
        # x = > torch.Size([32, 65, 384])
        x = self.norm(x)
        print('x=>', x.shape)
        # x = > torch.Size([32, 65, 384])
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # concat_f = > torch.Size([32, 768]
        print('concat_f=>', concat_f.shape)
        ret = self.cls_head_finetune(concat_f)
        return ret
