import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.TPM import TPM
from modeling.fusion_part.CRM import CRM
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        # No view
        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        cash_x = self.base(x, cam_label=cam_label, view_label=view_label)
        global_feat = cash_x[-1][:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cash_x, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return cash_x, feat
            else:
                return cash_x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class BASELINE(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BASELINE, self).__init__()
        self.NI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.TI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num

        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.mix_dim = 768
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.mix_dim = 512
        if 'deit' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.mix_dim = 384
        self.miss_type = cfg.TEST.MISS
        self.classifier = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.mix_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_score, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_score, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck(ori)
            ori_score = self.classifier(ori_global)
            if self.direct:
                return ori_score, ori
            else:
                return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck(ori)

            if self.neck_feat == 'after':
                pass
            else:
                ori_global = ori
            if self.test_feat == 0:
                return torch.cat([ori_global], dim=-1)


class TOPReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(TOPReID, self).__init__()
        self.NI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.TI = build_transformer(num_classes, cfg, camera_num, view_num, factory)
        self.RGB = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.num_head = 12
        self.mix_dim = 768
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.mix_dim = 512
            self.num_head = 8
        if 'deit' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.mix_dim = 384

        self.TPM = TPM(dim=self.mix_dim, num_heads=self.num_head)
        self.re = cfg.MODEL.RE
        if self.re:
            self.CRM = CRM(dim=self.mix_dim, num_heads=self.num_head, miss=cfg.TEST.MISS,
                           depth=cfg.MODEL.RE_LAYER)
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.layer = cfg.MODEL.LAYER
        self.direct = cfg.MODEL.DIRECT

        self.classifier_TPM = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        self.classifier_TPM.apply(weights_init_classifier)
        self.bottleneck_TPM = nn.BatchNorm1d(3 * self.mix_dim)
        self.bottleneck_TPM.bias.requires_grad_(False)
        self.bottleneck_TPM.apply(weights_init_kaiming)

        self.classifier_ViT = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        self.classifier_ViT.apply(weights_init_classifier)
        self.bottleneck_ViT = nn.BatchNorm1d(3 * self.mix_dim)
        self.bottleneck_ViT.bias.requires_grad_(False)
        self.bottleneck_ViT.apply(weights_init_kaiming)

        self.miss = cfg.TEST.MISS

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, cam_label=None, label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_score, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_score, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_score, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)

            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck_ViT(ori)
            ori_score = self.classifier_ViT(ori_global)

            TPM_feature = self.TPM(RGB_cash[self.layer], NI_cash[self.layer], TI_cash[self.layer])
            if self.re:
                loss_re = self.CRM(RGB_cash[self.layer], NI_cash[self.layer], TI_cash[self.layer])
            TPM_global = self.bottleneck_TPM(TPM_feature)
            TPM_score = self.classifier_TPM(TPM_global)
            if self.re:
                if self.direct:
                    return TPM_score, TPM_feature, ori_score, ori, loss_re
                else:
                    return TPM_score, TPM_feature, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, loss_re
            else:
                if self.direct:
                    return TPM_score, TPM_feature, ori_score, ori
                else:
                    return TPM_score, TPM_feature, RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.NI(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.TI(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.RGB(RGB, cam_label=cam_label, view_label=view_label)
            TPM_feature = self.TPM(RGB_cash[self.layer], NI_cash[self.layer], TI_cash[self.layer])
            if self.re:
                if self.miss == 'r':
                    RGB = self.CRM(ma=None, mb=NI_cash[self.layer], mc=TI_cash[self.layer])
                    TPM_feature = self.TPM(RGB, NI_cash[self.layer], TI_cash[self.layer])
                elif self.miss == "n":
                    NI = self.CRM(ma=RGB_cash[self.layer], mb=None, mc=TI_cash[self.layer])
                    TPM_feature = self.TPM(RGB_cash[self.layer], NI, TI_cash[self.layer])
                elif self.miss == 't':
                    TI = self.CRM(ma=RGB_cash[self.layer], mb=NI_cash[self.layer], mc=None)
                    TPM_feature = self.TPM(RGB_cash[self.layer], NI_cash[self.layer], TI)
                elif self.miss == 'rn':
                    RGB, NI = self.CRM(ma=None, mb=None, mc=TI_cash[self.layer])
                    TPM_feature = self.TPM(RGB, NI, TI_cash[self.layer])
                elif self.miss == 'rt':
                    RGB, TI = self.CRM(ma=None, mb=NI_cash[self.layer], mc=None)
                    TPM_feature = self.TPM(RGB, NI_cash[self.layer], TI)
                elif self.miss == 'nt':
                    NI, TI = self.CRM(ma=RGB_cash[self.layer], mb=None, mc=None)
                    TPM_feature = self.TPM(RGB_cash[self.layer], NI, TI)

            TPM_global = self.bottleneck_TPM(TPM_feature)
            if self.neck_feat == 'after':
                pass
            else:
                TPM_global = TPM_feature
            return torch.cat([TPM_global], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    if cfg.MODEL.BASE == 1:
        model = BASELINE(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building BASELINE===========')
    else:
        model = TOPReID(num_class, cfg, camera_num, view_num, __factory_T_type)
        print('===========Building TOPReID===========')
    return model
