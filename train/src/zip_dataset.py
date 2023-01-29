import random
import numpy as np
import torch


# https://www.kaggle.com/code/werus23/g2net-pytorch-with-generated-data/notebook#Dataset
class ZipDataset(torch.utils.data.Dataset):
    """
    img, y = dataset[i]
      img (np.float32): 2 x 360 x 180
      y (np.float32): label 0 or 1
    """
    def __init__(self, pos_len=5000, neg_len=9900, path=f'../g2net-generated-signals/archive',mod= 100, noise = 0.99,
                 transforms=None):
        self.path = path
        self.mod = int(mod)
        self.noise = noise
        self.noise_type = 2
        
        self.pos_len = pos_len
        self.neg_len = neg_len
        self.len = pos_len + neg_len
        
        self.mixup = True
        self.mixup_prob = 0.1
        self.perm_pos = np.random.permutation(np.arange(self.pos_len))
        self.perm_neg = np.random.permutation(np.arange(self.neg_len))
        
        self.transforms = transforms
    
    def __len__(self):
        return self.len
    
    def gen_noise(self, shape):
        ns = 0.15
        nr = 0.05

        noise_shape = (360*4140)
        noise_L_r = self.gen_noise(noise_shape)
        noise_H_r = noise_L_r*(1-ns)+self.gen_noise(noise_shape)*ns
        noise_L_i = noise_L_r*(1-nr)+self.gen_noise(noise_shape)*nr
        noise_H_i = noise_H_r*(1-nr)+self.gen_noise(noise_shape)*nr

        noise_r = np.stack([noise_L_r,noise_H_r]) *1e22
        noise_i = np.stack([noise_L_i,noise_H_i]) *1e22
        img_n = noise_r**2 + noise_i**2
        return img_n
    
    def get_negative(self,i):
        file_name = f'{self.path}/0_data_{self.mod*(1+(i)//self.mod)}/signals_{i%self.mod}.npy'
        img = np.load(file_name).astype(np.float64)
        y=0.0
        return img, y
    
    def get_positive(self, i):
        file_name = f'{self.path}/1_data_{self.mod*(1+(i)//self.mod)}/signals_{i%self.mod}.npy'
        img = np.load(file_name).astype(np.float64)
        
        noise_id = int(random.random()*self.neg_len)
        noise_r = random.random()*0.05+0.95
        img = (np.sqrt(img)*(1-noise_r)+np.sqrt(self.get_negative(noise_id)[0])*noise_r)**2
        y=1.0
        return img, y
    
    def get_noise(self):
        return self.noise
    
    def get_mixup(self, i, t):
        if t==1:
            mix_img = (self.get_positive(i)[0] + self.get_positive(self.perm_pos[i])[0])/2
            if random.random() < 1/self.pos_len:
                self.pos_perm = np.random.permutation(np.arange(self.pos_len))
        else:
            mix_img = (self.get_negative(i)[0] + self.get_negative(self.perm_neg[i])[0])/2
            if random.random() < 1/self.neg_len:
                self.neg_perm = np.random.permutation(np.arange(self.neg_len))
        return mix_img, t
    
    def __getitem__(self, i):
        if i<self.pos_len:
            if self.mixup and random.random() < self.mixup_prob:
                img, y = self.get_mixup(i,1)
            else:
                img, y = self.get_positive(i)
        else:
            i = i-self.pos_len
            if self.mixup and random.random() < self.mixup_prob:
                img, y = self.get_mixup(i,0)
            else:
                img, y = self.get_negative(i)
        img = ((img)/img.mean() ).astype(np.float32)
        
        if self.transforms:
            img = img.transpose(1, 2, 0)  # (H,W,C)
            img = self.transforms(image=img)["image"]
        
        return img, y
    
