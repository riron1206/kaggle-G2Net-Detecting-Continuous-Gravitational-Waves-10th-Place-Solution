import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/pfnet-research/kaggle-alaska2-3rd-place-solution/blob/01d302f9897ea04baf46f2fe8585055a5ecaad69/alaska2/models.py#L12
def patch_first_conv_stride(model, first_stride=(1,1)):
    """
    最初の conv レイヤーのモデル ストライドをfirst_strideに変更
    基本的に画像解像度をさらに拡大するためにコンピューター ビジョン競技で使用される一般的なトリック
    モデルはバックボーンを介してより高い解像度でトレーニングできます
    Christof と Philipp は、ALASKA2 Steganalysis コンテストで同じトリックを使用
    https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168870
    """
    # decrease first conv's stride
    modules_iter = iter(model.modules())
    for module in modules_iter:
        if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):  # eff_netは最初の層(conv_stem)はstride=(2, 2)
            break
    module.stride = first_stride
    return model

### patch_first_conv_stride test
#import timm
#import torch
#m = timm.create_model("tf_efficientnet_b5_ap", pretrained=False,
#                      num_classes=1,
#                      in_chans=2)
#m = patch_first_conv_stride(m)
#print(m)  # 最初の conv レイヤーのストライド(1,1)になってる
#from torchinfo import summary
#summary(
#    m.to("cpu"),
#    input_size=(2, 2, 224, 224),
#    col_names=["output_size", "num_params"],
#)


# ==============================================================
# GeM Pooling
# 画像検索の精度を向上させる、トレーニング可能な一般化平均(GeM)Pooling層を提案している。この層は、MaxPoolingやAveragePoolingの一般化と捉えられる
# 日本語の解説: https://zenn.dev/takoroy/scraps/151d11817e3700
#
# ![](https://i.imgur.com/thTgYWG.jpg)
# ==============================================================
# https://amaarora.github.io/2020/08/30/gempool.html
# https://github.com/Fkaneko/kaggle_g2net_gravitational_wave_detection/blob/8bb32cc675e6b56171da8a3754fffeda41e934bb/src/modeling/model_arch/conv_models.py#L34
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=True):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=requires_grad)  # requires_grad=Falseならパラメータ学習しない（https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/discussion/275431 よりFalseがworkしたらしい）
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'




