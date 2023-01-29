import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from time import time

# https://www.kaggle.com/code/anonamename/g2net2-mean-std-calc-ma-img4224
def calc_mean_std_max(dataset, full_loader, N_CHANNELS=2):
    """
    dataloaderから平均値, 標準偏差, 最大値を計算
    https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    """    
    x_max = torch.zeros(N_CHANNELS)
    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)
    print('==> Computing mean and std..')
    for xxx in tqdm(full_loader):
        inputs = xxx[0]
        #print(inputs.shape)  # torch.Size([1, 360, 4096, 2])。batch_size=1の必要あり
        for i in range(N_CHANNELS):
            mean[i] += inputs[:,:,:,i].mean()
            std[i] += inputs[:,:,:,i].std()
            
            # 最大値も残す
            _max = inputs[:,:,:,i].max()
            if _max > x_max[i]:
                x_max[i] = _max
            
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean, std, max:", mean.numpy(), std.numpy(), x_max.numpy())
    return mean.numpy(), std.numpy(), x_max.numpy()