import torch
import torch.nn as nn
import cv2
import os 
import albumentations as A
import numpy as np
from osgeo import gdal
from albumentations.pytorch import ToTensorV2
from LECOS import LECOS

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

target_size = 320
n_class = 5
n_bands = 7

data_path = ""
ckpt_path = ""
mask_path = data_path + "/inference/"

if not os.path.exists(mask_path):
    os.makedirs(mask_path)
set_name = "test"

transform = A.Compose([ 
            A.Resize(target_size,target_size,interpolation=cv2.INTER_CUBIC),
            ToTensorV2(),
        ])
        
outer_dim = 128
inner_dim = 12
outer_head = 4
inner_head = 2
configs = {
    'depths': [2, 10, 6, 2],
    'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
    'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
    'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
    'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    'conv_dim': 128
}
model = LECOS(configs=configs, in_chans=n_bands, img_size=target_size, num_classes=n_class)

####=================================================================#### 
####                        1.Sentinel2-GDGQ                         ####
####=================================================================#### 
with open(data_path + '/' +  set_name + ".txt", "r") as f:
    testImageList = []
    for line in f.readlines():
        fname = line.strip('\n')
        image = data_path + "/patch/" + fname
        testImageList.append(image)
####=================================================================#### 

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
# ckpt = torch.load("Code/SubpixelMapping/MultiFramework/check points/Sentinel2-GDGQ/PyramidTNT M/wm 300/0.0001/ckpt_ep:255.pth", map_location=torch.device('cpu'))
if torch.cuda.device_count() > 1:
    # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。
    model = nn.DataParallel(model) 
model.load_state_dict(ckpt["model"]) # 接着就可以将模型参数load进模型
model = model.cuda()

model.eval()
lb_list = []
with torch.no_grad():
    for i in range(0, len(testImageList)):
        img = gdal.Open(testImageList[i].split(".")[0]).ReadAsArray()
        img = np.transpose(img, (1,2,0))
        file_name = testImageList[i].split("/")[-1].split(".")[0]

        ####=================================================================####
        transform = transform(image=img)
        img = transform["image"]
        img = torch.unsqueeze(img,0)

        pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1).cpu().numpy().squeeze()
        lb_mask = np.zeros((pred.shape[0],pred.shape[1],3)) + 255
        pred = pred + 1
        
        key1 = pred==1
        key2 = pred==2
        key3 = pred==3
        key4 = pred==4
        key5 = pred==5
        key6 = pred==6

        lb_mask[:,:,2][key1] = 252
        lb_mask[:,:,1][key1] = 250
        lb_mask[:,:,0][key1] = 205

        lb_mask[:,:,2][key2] = 0
        lb_mask[:,:,1][key2] = 123
        lb_mask[:,:,0][key2] = 79

        lb_mask[:,:,2][key3] = 157
        lb_mask[:,:,1][key3] = 221
        lb_mask[:,:,0][key3] = 106

        lb_mask[:,:,2][key4] = 10
        lb_mask[:,:,1][key4] = 78
        lb_mask[:,:,0][key4] = 151

        lb_mask[:,:,2][key5] = 155
        lb_mask[:,:,1][key5] = 36
        lb_mask[:,:,0][key5] = 22

        lb_mask[:,:,2][key6] = 0
        lb_mask[:,:,1][key6] = 0
        lb_mask[:,:,0][key6] = 0

        lb_mask = lb_mask.astype(np.uint8)
        cv2.imwrite(mask_path+file_name+".png",lb_mask)