import torch
import torch.nn as nn
import timm
from collections import defaultdict, Counter, OrderedDict

from src import largekernel
from src import layer_edit


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=cfg.num_classes,
                                     in_chans=cfg.ch)
    def forward(self, x, labels=None):
        return self.net(x)[:,0]


class CustomModelFreq(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y2 = self.head2(feat)
        return y2


class CustomModelFreqStride(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)

        # 最初の conv レイヤーのモデル ストライドをfirst_strideに変更
        self.net = layer_edit.patch_first_conv_stride(self.net, first_stride=cfg.first_stride)

        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y2 = self.head2(feat)
        return y2


# https://www.kaggle.com/code/hirune924/2ndplace-solution/notebook を参考にdropout入れて過学習避ける
class CustomModelDrop(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=cfg.num_classes,
                                     in_chans=cfg.ch,
                                     drop_rate=cfg.drop_rate,  # headのドロップアウト率 (デフォルト: 0)
                                     drop_path_rate=cfg.drop_path_rate  # 中間層のドロップアウト率 (デフォルト: 0)
                                    )
    def forward(self, x, labels=None):
        return self.net(x)[:,0]


class CustomModelMultiOutput(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)
        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelMultiOutputDrop(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch,
                                     drop_rate=cfg.drop_rate,  # headのドロップアウト率 (デフォルト: 0)
                                     drop_path_rate=cfg.drop_path_rate  # 中間層のドロップアウト率 (デフォルト: 0)
                                    )
        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelMultiOutput2(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)
        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
        # リアルデータかシュミレーションデータかの分類ヘッド
        self.head3 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, 1)
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        y3 = self.head3(feat)
        return y1[:,0], y2, y3[:,0]


class CustomModelMultiOutput3(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)
        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
        # h0deg//10の分類ヘッド
        self.head3 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, 11)
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        y3 = self.head3(feat)
        return y1[:,0], y2, y3


class CustomModelMultiOutputLargeKernelNoAvgPool(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()

        conv_in_chans = 32  # eff_b0
        #_C, _H, _W = 16, 31, 255
        _C, _H, _W = 16, 31, cfg.size_w

        if "b4" in cfg.model_name:
            conv_in_chans = 48
            _C = conv_in_chans//2
        elif "b5" in cfg.model_name:
            conv_in_chans = 48
            _C = conv_in_chans//2

        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=conv_in_chans,
                                    )
        self.net.conv_stem = nn.Sequential(
            nn.Identity(),
            largekernel.LargeKernel_debias(1, _C, [_H, _W], 1, [_H//2, _W//2], 1, 1, False),
            self.net.conv_stem,
            )

        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )

    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelMultiOutputGeM(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     global_pool='',
                                     in_chans=cfg.ch)

        self.pooling = layer_edit.GeM(p=cfg.gem_p)

        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        feat = self.pooling(feat).flatten(1)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelMultiOutputStride(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)

        # 最初の conv レイヤーのモデル ストライドをfirst_strideに変更
        self.net = layer_edit.patch_first_conv_stride(self.net, first_stride=cfg.first_stride)

        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelMultiOutputStrideGeM(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     global_pool='',
                                     in_chans=cfg.ch)

        # 最初の conv レイヤーのモデル ストライドをfirst_strideに変更
        self.net = layer_edit.patch_first_conv_stride(self.net, first_stride=cfg.first_stride)

        self.pooling = layer_edit.GeM(p=cfg.gem_p)

        self.head1 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )
        self.head2 = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, (500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1)  # 40-500Hz // cfg.freq_div_n
        )
    def forward(self, x, labels=None):
        feat = self.net(x)
        feat = self.pooling(feat).flatten(1)
        y1 = self.head1(feat)
        y2 = self.head2(feat)
        return y1[:,0], y2


class CustomModelAddEmb(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=0,
                                     in_chans=cfg.ch)
        self.head = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features+(500 // cfg.freq_div_n) - (40 // cfg.freq_div_n) + 1, cfg.num_classes)  # embを1次元増やす
        )
    def forward(self, x, add_x):
        feat = self.net(x)
        #print(feat.shape, add_x.shape)
        y = self.head( torch.cat((feat, add_x), 1) )
        return y[:,0]


class CustomModelMultiInput(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.net1 = timm.create_model(cfg.model_name, pretrained=pretrained,
                                      num_classes=0,
                                      in_chans=cfg.ch//2)
        self.net2 = timm.create_model(cfg.model_name, pretrained=pretrained,
                                      num_classes=0,
                                      in_chans=cfg.ch//2)
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net1.num_features + self.net2.num_features, cfg.num_classes)
        )

    def forward(self, x, labels=None):

        x1 = x[:, :cfg.ch//2, :, :]
        x2 = x[:, cfg.ch//2:, :, :]
        #print(x1.shape, x2.shape)

        feat1 = self.net1(x1)
        feat2 = self.net2(x2)
        y = self.head( torch.cat((feat1, feat2), 1) )
        #print(feat1.shape, feat2.shape, y.shape)

        return y[:,0]


class CustomModelLargeKernel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        if pretrained:
            pth = "/volume/kaggle/g2net2/kaggle_dl/g2net-detecting-continuous-gravitational-waves-v0/model_best.pth"
        else:
            pth = ""
        self.net = largekernel.get_model(
            model_name="tf_efficientnetv2_b0",
            path=pth,
            conv_in_chans=32
        )
        self.net.classifier = nn.Sequential(
            #nn.Dropout(p=0.3),
            #nn.BatchNorm1d(self.net1.num_features + self.net2.num_features),
            nn.Linear(self.net.num_features, cfg.num_classes)
        )

    def forward(self, x, labels=None):
        return self.net(x)[:,0]


class CustomModelLargeKernelNoAvgPool(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()

        conv_in_chans = 32  # eff_b0
        #_C, _H, _W = 16, 31, 255
        _C, _H, _W = 16, 31, cfg.size_w

        if "b4" in cfg.model_name:
            conv_in_chans = 48
            _C = conv_in_chans//2
        elif "b5" in cfg.model_name:
            conv_in_chans = 48
            _C = conv_in_chans//2

        self.net = timm.create_model(cfg.model_name, pretrained=pretrained,
                                     num_classes=cfg.num_classes,
                                     in_chans=conv_in_chans,
                                    )
        self.net.conv_stem = nn.Sequential(
            nn.Identity(),
            largekernel.LargeKernel_debias(1, _C, [_H, _W], 1, [_H//2, _W//2], 1, 1, False),
            self.net.conv_stem,
            )

    def forward(self, x, labels=None):
        return self.net(x)[:,0]


# https://github.com/sinpcw/kaggle-whale2/blob/master/models.py
def loadpth(pth: str, map_location=None) -> OrderedDict:
    """
    パラメータロードのヘルパー関数.
    DataParallel化したモデルは module.xxxx という形式で保存されるため読込み時にmodule.から始まる場合はそれを取除く.
    """
    ostate = torch.load(pth, map_location=map_location)['model']
    nstate = OrderedDict()
    for k, v in ostate.items():
        if k.startswith('module.'):
            nstate[k[len('module.'):]] = v
        else:
            nstate[k] = v
    return nstate


def load_state_dict_skip_missmatch(model, state, strict=True):
    """
    sizeが合わない層はloadしないload_state_dict
    https://github.com/PyTorchLightning/pytorch-lightning/issues/4690
    """
    model_state = model.state_dict()
    is_changed = False
    for k in state:
        if k in model_state:
            if state[k].shape != model_state[k].shape:
                print(
                    f"Skip loading parameter: {k}, "
                    f"required shape: {model_state[k].shape}, "
                    f"loaded shape: {state[k].shape}"
                )
                state[k] = model_state[k]
        else:
            print(f"Dropping parameter {k}")

    model.load_state_dict(state, strict=strict)

    return model


### test
#print(CFG.model_name)
#m = CustomModelMultiOutputStrideGeM(pretrained=CFG.pretrained).cuda()
##m = convert_model(m).cuda() # Batch NormをSync Batch Normに変換
##m = DataParallelWithCallback(m, device_ids=CFG.device_ids) # Data Parallel
#x = torch.rand(CFG.batch_size, 2, CFG.size_h, CFG.size_w)
#o1, o2 = m(x.cuda())
#print(o1, o1.shape)
#print(o2, o2.shape)
#m
##from torchinfo import summary
##summary(
##    m.to(device),
##    input_size=(2, 2, CFG.size_h, CFG.size_w),
##    col_names=["output_size", "num_params"],
##)