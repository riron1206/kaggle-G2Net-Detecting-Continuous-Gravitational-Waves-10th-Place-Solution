# https://www.kaggle.com/code/laeyoung/g2net-large-kernel-inference/notebook
import gc, glob, os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from timm import create_model
from sklearn.preprocessing import RobustScaler


# ====================================================================================
# Data Preprocess
# ====================================================================================
def normalize(X, is_robustscaler=False):
    """
    外れ値を除いて振幅をpowerに変換する
    Args
        X: 複素数の振幅
    Returns
        X: float型のpower
    """
    X = X[..., None]  # 振幅の次元増やす。shape: (360, 4721) -> (360, 4721, 1)
    X = X.view(X.real.dtype)  # 振幅の実数の型であるfloat64にして、実数と虚数を別次元にする。shape: (360, 4721, 1) -> (360, 4721, 2)
    X = X ** 2  # 実数と虚数をそれぞれ2乗
    X = X.sum(-1)  # 2乗した振幅の合計を取ってパワーにする。shape: (360, 4721)
    if is_robustscaler == False:
        POS = int(X.size * 0.99903)  # パワーの全要素数(360 x 4721)に係数0.99903掛ける。0.99903番目のデータを正規化の値として後続処理で使う
        _a = (POS + 0.4) / (X.size + 0.215)  # _aは0.9999とか
        EXP = norm.ppf(_a)  # _aの分位数の標準正規分布の値を取得。3.09とか。stats.norm.ppf(0.25)は第１四分位点、ppf(0.5)は中央値をとる。locとscaleを省略すると標準正規分布
        scale = np.partition(X.flatten(), POS, -1)[POS]  # np.partition: 値の大きさから見た順番がN番目の要素で内容を仕切る
        X /= scale / EXP.astype(scale.dtype) ** 2  # 正規化してるようだが式の意味は不明...
    else:
        # 追加
        X *= 1e44
        X = RobustScaler().fit_transform(X)
    return X

def dataload(filepath, is_robustscaler=False, is_norm=True):
    """
    h5pyファイルロードして file_id, power, H1のpowerの平均値, L1のpowerの平均値 を返す
    H1, L1のタイムスタンプが揃うようにgap部分はnp.nanを入れる
    Args
        filepath: h5pyファイルのパス
    Returns
        fid: file_id
        astime: 振幅から出したpower
        H1.mean(): H1のpowerの平均値
        L1.mean(): L1のpowerの平均値
    """
    if is_norm:
        astime = np.full([2, 360, 5760], np.nan, dtype=np.float32)
    else:
        astime = np.full([2, 360, 5760], np.nan, dtype=np.complex128)
    with h5py.File(filepath, "r") as f:
        fid, _ = os.path.splitext(os.path.split(filepath)[1])
        # 1800で割るのでタイムポイントのインデックスになる
        # [10, 11, 15, ...]みたいなの。gapがあると連番のインデックスではなくなる
        HT = (np.asarray(f[fid]["H1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        LT = (np.asarray(f[fid]["L1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        # H1,L1の最小のタイムポイントを取得
        MIN = min(HT.min(), LT.min())
        # 最小のタイムポイントで引いて開始時刻を揃える
        HT -= MIN
        LT -= MIN
        # 振幅を正規化
        if is_norm:
            H1 = normalize(np.asarray(f[fid]["H1"]["SFTs"], np.complex128), is_robustscaler=is_robustscaler)  # 正規化してパワーに変換
        else:
            H1 = np.asarray(f[fid]["H1"]["SFTs"], np.complex128)  # 複素数の振幅のまま
        valid = HT < 5760  # データの数を5760までにする
        astime[0][:, HT[valid]] = H1[:, valid]  # gapを加味した配列に変更する。gap部分はnp.nanになる
        if is_norm:
            L1 = normalize(np.asarray(f[fid]["L1"]["SFTs"], np.complex128), is_robustscaler=is_robustscaler)  # 正規化してパワーに変換
        else:
            L1 = np.asarray(f[fid]["L1"]["SFTs"], np.complex128)  # 複素数の振幅のまま
        valid = LT < 5760  # データの数を5760までにする
        astime[1][:, LT[valid]] = L1[:, valid]  # gapを加味した配列に変更する。gap部分はnp.nanになる
    gc.collect()
    return fid, astime, H1.mean(), L1.mean()

# 追加
def dataload_hstack_normalize(filepath, is_robustscaler=False):
    """
    h5pyファイルロードして file_id, power, H1のpowerの平均値, L1のpowerの平均値 を返す
    H1, L1のタイムスタンプが揃うようにgap部分はnp.nanを入れる
    各チャネルを横に連結して正規化
    Args
        filepath: h5pyファイルのパス
    Returns
        fid: file_id
        astime: 振幅から出したpower
        H1.mean(): H1のpowerの平均値
        L1.mean(): L1のpowerの平均値
    """
    astime = np.full([2, 360, 5760], np.nan, dtype=np.float32)
    with h5py.File(filepath, "r") as f:
        fid, _ = os.path.splitext(os.path.split(filepath)[1])
        # 1800で割るのでタイムポイントのインデックスになる
        # [10, 11, 15, ...]みたいなの。gapがあると連番のインデックスではなくなる
        HT = (np.asarray(f[fid]["H1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        LT = (np.asarray(f[fid]["L1"]["timestamps_GPS"]) / 1800).round().astype(np.int64)
        # H1,L1の最小のタイムポイントを取得
        MIN = min(HT.min(), LT.min())
        # 最小のタイムポイントで引いて開始時刻を揃える
        HT -= MIN
        LT -= MIN

        # 各チャネルを横に連結
        H1 = np.asarray(f[fid]["H1"]["SFTs"], np.complex128)
        L1 = np.asarray(f[fid]["L1"]["SFTs"], np.complex128)
        h1_l1_amp = np.hstack((H1, L1))
        #print("H1, L1, h1_l1_amp shape:", H1.shape, L1.shape, h1_l1_amp.shape)

        # 振幅を正規化
        h1_l1_amp = normalize(h1_l1_amp, is_robustscaler=is_robustscaler)

        # 各チャネルをもとに戻す
        H1 = h1_l1_amp[:,:H1.shape[1]]
        L1 = h1_l1_amp[:,H1.shape[1]:]

        valid = HT < 5760  # データの数を5760までにする
        astime[0][:, HT[valid]] = H1[:, valid]  # gapを加味した配列に変更する。gap部分はnp.nanになる
        L1 = normalize(np.asarray(f[fid]["L1"]["SFTs"], np.complex128), is_robustscaler=is_robustscaler)
        valid = LT < 5760  # データの数を5760までにする
        astime[1][:, LT[valid]] = L1[:, valid]  # gapを加味した配列に変更する。gap部分はnp.nanになる

    gc.collect()
    return fid, astime, H1.mean(), L1.mean()

def preprocess(num, input, h1_m, l1_m, is_cuda=False, is_tta=False):
    """
    タイムスタンプが飛んでるgap部分はH1, L1の平均値にノイズを掛けた値にする前処理
    Args:
        num: batch_size
        input: 振幅をpowerにしたarray。gap部分はnp.nanになっている。shape: (num, 2, 360, 5760)
        h1_m: H1のpowerの平均値
        l1_m: L1のpowerの平均値
    Returns
        tta: タイムスタンプが飛んでるgap部分はH1, L1の平均値にノイズを掛けた値になったpower。shape: (num, 2, 360, 5760)
    """
    # H1, L1の平均値にノイズを掛けた行列用意。
    if is_cuda:
        input = torch.from_numpy(input).to("cuda", non_blocking=True)  # non_blockingは非同期処理するoption
        rescale = torch.tensor([[H1, L1]]).to("cuda", non_blocking=True)  # non_blockingは非同期処理するoption
    else:
        input = torch.from_numpy(input)
        rescale = torch.tensor([[h1_m, l1_m]])
    if is_tta:
        tta = (
            torch.randn(
                [num, *input.shape, 2], device=input.device, dtype=torch.float32
            )
            .square_()  # 各要素を二乗
            .sum(-1)  # 最後の次元を基準にsum取る
        )  # shape: [num, 2, 360, 5760])
        tta *= rescale[..., None, None] / 2  # H1, L1の平均値にノイズを掛ける
        valid = ~torch.isnan(input)  # np.nanのgap部分以外のデータだけ取り出し
        tta[:, valid] = input[valid].float()  # 取り出したgap部分以外のデータは実際のpowerの値を入れて、gap部分だけがH1, L1の平均値にノイズ掛けた値にする
    else:
        tta = torch.ones(num, *input.shape)
        tta *= rescale[..., None, None] / 2  # H1, L1の平均値を掛ける
        valid = ~torch.isnan(input)  # np.nanのgap部分以外のデータだけ取り出し
        tta[:, valid] = input[valid].float()  # 取り出したgap部分以外のデータは実際のpowerの値を入れて、gap部分だけがH1, L1の平均値を掛けた値にする

    return tta

def preprocess_v2(num, input, h1_m, l1_m, is_cuda=False, is_tta=False):
    """
    タイムスタンプが飛んでるgap部分はH1, L1の前後の値の平均値にする前処理
    Args:
        num: batch_size
        input: 振幅をpowerにしたarray。gap部分はnp.nanになっている。shape: (num, 2, 360, 5760)
        h1_m: H1のpowerの平均値
        l1_m: L1のpowerの平均値
    Returns
        tta: タイムスタンプが飛んでるgap部分はH1, L1の平均値にノイズを掛けた値になったpower。shape: (num, 2, 360, 5760)
    """
    
    # 前後の値から平均値で補完。axis=1で列方向で補完。limit_direction='both'で両端も補完
    input[0] = pd.DataFrame(input[0], dtype=np.float64).interpolate(axis=1, limit_direction='both').values
    input[1] = pd.DataFrame(input[1], dtype=np.float64).interpolate(axis=1, limit_direction='both').values
    
    # H1, L1の平均値にノイズを掛けた行列用意。
    if is_cuda:
        input = torch.from_numpy(input).to("cuda", non_blocking=True)  # non_blockingは非同期処理するoption
        rescale = torch.tensor([[H1, L1]]).to("cuda", non_blocking=True)  # non_blockingは非同期処理するoption
    else:
        input = torch.from_numpy(input)
        rescale = torch.tensor([[h1_m, l1_m]])
    if is_tta:
        tta = (
            torch.randn(
                [num, *input.shape, 2], device=input.device, dtype=torch.float32
            )
            .square_()  # 各要素を二乗
            .sum(-1)  # 最後の次元を基準にsum取る
        )  # shape: [num, 2, 360, 5760])
        tta *= rescale[..., None, None] / 2  # H1, L1の平均値にノイズを掛ける
        valid = ~torch.isnan(input)  # np.nanのgap部分以外のデータだけ取り出し
        #print("#### valid.shape:", valid.shape)
        tta[:, valid] = input[valid].float()  # 取り出したgap部分以外のデータは実際のpowerの値を入れて、gap部分だけがH1, L1の平均値にノイズ掛けた値にする
    else:
        tta = torch.ones(num, *input.shape)
        tta *= rescale[..., None, None] / 2  # H1, L1の平均値を掛ける
        valid = ~torch.isnan(input)  # np.nanのgap部分以外のデータだけ取り出し
        tta[:, valid] = input[valid].float()  # 取り出したgap部分以外のデータは実際のpowerの値を入れて、gap部分だけがH1, L1の平均値を掛けた値にする

    return tta

def preprocess_none(num, input, h1_m, l1_m, is_cuda=False, is_tta=False):
    """
    前処理なし
    Args:
        num: batch_size
        input: 振幅をpowerにしたarray。gap部分はnp.nanになっている。shape: (num, 2, 360, 5760)
        h1_m: H1のpowerの平均値
        l1_m: L1のpowerの平均値
    Returns
        tta: タイムスタンプが飛んでるgap部分はH1, L1の平均値にノイズを掛けた値になったpower。shape: (num, 2, 360, 5760)
    """
    # H1, L1の平均値にノイズを掛けた行列用意。
    if is_cuda:
        input = torch.from_numpy(input).to("cuda", non_blocking=True)  # non_blockingは非同期処理するoption
    else:
        input = torch.from_numpy(input)
    return input.reshape(num, *input.shape)

# ====================================================================================
# Model
# ====================================================================================
class LargeKernel_debias(nn.Conv2d):
    """
    一般的な畳み込みニューラルネットワーク(CNN)では、3×3のような小さなカーネルを積み重ねることで大きな受容野を構築
    一方、ViTでは、Multi-Head Self-Sttention(MHSA)により、単一の層のみでも大きな受容野が実現
    CNNをViTに近づけるために、多数の小さなカーネルで大きな受容野(large receptive fields)を実現する既存のCNNの代わりに、少数の大きなカーネルを使用する
    31×31という、一般的なCNNと比べて大きいカーネルサイズを利用する
    https://ai-scholar.tech/articles/treatise/large_kernel
    """
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)
        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2 + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)
        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        #output.sub_(power.mul_(ratio))
        torch.sub( output, torch.mul(power, ratio) )
        return output.unflatten(0, input.shape[:2]).flatten(1, 2)

def get_model(model_name="tf_efficientnetv2_b0",
              path="/volume/kaggle/g2net2/kaggle_dl/g2net-detecting-continuous-gravitational-waves-v0/model_best.pth",
              conv_in_chans=32
             ):
    model = create_model(
        model_name,
        in_chans=conv_in_chans,  # tf_efficientnetv2_b0 のmodel.conv_stem の入力チャネルである32にしているみたい。実際の入力チャネルは2
        num_classes=2,
    )
    if path != "":
        state_dict = torch.load(path)
        C, _, H, W = state_dict["conv_stem.2.weight"].shape
        #print(C,H,W)  # 16, 31, 255
    else:
        # Cはmodel.conv_stem の入力チャネルの1/2に合わせないとエラーになる
        # model.conv_stem はefficientnet の最初の層。model.conv_stem の入力チャネルはモデルによってサイズ違う
        # eff_b0はC=16, eff_b4とeff_b5はC=24
        # Hはカーネルの縦のサイズになるので31x31のままで変更しないほうが良さそう
        #
        C, H, W = 16, 31, 255
    model.conv_stem = nn.Sequential(
        nn.Identity(),
        # nn.AvgPool2dの最初の3引数は
        # kernel_size（平均するフィルターサイズ。(1,9)なら縦1ピクセル横9ピクセルのフィルター。conv2dの場合はフィルターの重みとピクセルの値で内積=畳み込みするが、AvgPoolだから単純にこのサイズ単位でピクセルの値を平均する）,
        # stride（フィルターをずらす長さ）,
        # padding（パディングするサイズ。処理前に入力データの周囲に固定値を入れて出力サイズを調整する。(0,4)なら左右4ピクセル、要は8ピクセル伸びた画像で平均する）
        # count_include_pad はTrue の場合、AvgPoolの平均計算にゼロパディングが含まれる
        nn.AvgPool2d((1, 9), (1, 8), (0, 4), count_include_pad=False),  # 単純に画像の移動平均を取ってるだけ。要は画像の横幅を5760->720に圧縮をするためにAvgPool2dを使って前処理してる
        LargeKernel_debias(1, C, [H, W], 1, [H//2, W//2], 1, 1, False),
        model.conv_stem,
    )
    if path != "":
        model.load_state_dict(state_dict)
    return model


# test
#fid, power, h1_m, l1_m = dataload("/volume-ssd/kaggle_g2net2/g2net-detecting-continuous-gravitational-waves/test/698567d90.hdf5")
#tta = preprocess(1, power, h1_m, l1_m, is_cuda=False)
#print(tta.shape)
#model = get_model()
#model.to("cpu")
#model.eval()
#with torch.no_grad():
#    yhat = model(tta)
#print(yhat)
