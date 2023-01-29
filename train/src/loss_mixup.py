import torch
import numpy as np

def x_mixup(x, y, a: float = 1.0, enable: bool = True):
    """
    lossでmixupするために特徴量をランダムに alpha blend する
    ラベルはblendしないで、この関数呼び出した後の処理で、特徴量のalphaと同じ量lossをblendする
    Arg:
        x: 特徴量のbatch。Dataloaderから出てきたxを想定。0~1の正規化済みの値を想定
        y: ラベルのbatch。Dataloaderから出てきたyを想定。onehotかどうか関係なくいけるはず
        a: alpha blendのalpha
        enable: mixupするかのフラグ。Trueならランダムに実行。Falseなら必ず実行しない
    Usage:
        %reload_ext autoreload
        %autoreload 2
        from src.loss_mixup import x_mixup
        ...
        # train_fn()内にて
        fx, t1, t2, a, usemix = x_mixup(f, t, a=0.4, enable=CFG.is_loss_mixup)
        if usemix:
            fx = fx.to(device)
            t1 = t1.to(device).long()
            t2 = t2.to(device).long()

            if CFG.apex:
                with autocast():
                    y1 = model(fx, t1)
                    y2 = model(fx, t2)
                    loss = a * criterion(y1, t1) + (1.0 - a) * criterion(y2, t2)
            else:
                y1 = model(fx, t1)
                y2 = model(fx, t2)
                loss = a * criterion(y1, t1) + (1.0 - a) * criterion(y2, t2)
        else:
            fx = fx.to(device)
            t1 = t1.to(device).long()

            if CFG.apex:
                with autocast():
                    y1 = model(fx, t1)
                    loss = criterion(y1, t1)
            else:
                y1 = model(fx, t1)
                loss = criterion(y1, t1)
    """
    a = np.clip(a, 0.0, 1.0)
    if enable and np.random.rand() >= 0.5:
        j = torch.randperm(x.size(0))
        u = x[j]
        z = y[j]
        a = np.random.beta(a, a)
        w = a * x + (1.0 - a) * u
        return w, y, z, a, True
    return x, y, y, 1.0, False
