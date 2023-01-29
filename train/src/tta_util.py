import torch

def tta_tensor(img, ops):
    # input: NxCxHxW
    if ops == 0:
        pass
    elif ops == 1:
        img = torch.flip(img, [-1])  # hflip
    elif ops == 2:
        img = torch.flip(img, [-2])  # vflip
    elif ops == 3:
        img = torch.flip(img, [-1, -2])  # hflip+vflip
    elif ops == 4:
        img = torch.rot90(img, 1, [2, 3])  # 90x1=90度回転
    elif ops == 5:
        img = torch.rot90(img, 3, [2, 3])  # 90x3=270度回転
    elif ops == 6:
        img[:,0,:,:] = 0.0  # ch1 drop
    elif ops == 7:
        img[:,1,:,:] = 0.0  # ch2 drop
    elif ops == 8:
        # ch12 swap
        i0 = img[:,0,:,:].clone()
        img[:,0,:,:] = img[:,1,:,:]
        img[:,1,:,:] = i0
    else:
        pass
    return img


#### test
#tta_ops = [0, 1, 2]  # orig, hflip, vflip
#print("tta_ops:", tta_ops)
#
#preds = []
#for i, (images, _, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
#    images = images.to(device)
#    outputs = None
#
#    with torch.no_grad():
#        for _ops in tta_ops:
#            o, _ = model( tta_tensor(images, _ops) )  # for MultiOutput
#            if outputs is None:
#                outputs = o
#            else:
#                outputs += o
#    outputs /= len(tta_ops)
#    # for BCE
#    preds.append(outputs.sigmoid().cpu().detach().numpy())