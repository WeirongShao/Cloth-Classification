
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from resnet import res2net50
from dataloader import get_loader
from utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import gc
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def structure_loss(pred_y, target):
    # print(pred_y,target)
    loss1 = F.cross_entropy(pred_y, target.long())
    return loss1

from tqdm import tqdm
def train(train_loader, val_loader, model, optimizer):

    model.eval()
    # ---- multi-scale training ----
    with torch.no_grad():
        loss_record = AvgMeter()
        logit_result = []
        logit_data =  []
        label_list = []
        for i, (packimage,packlabel) in enumerate(tqdm(val_loader)):

            # ---- data prepare ----
            images = packimage
            label = packlabel
            #print(len(images))
            images = Variable(images).to(device)
            label = Variable(label).to(device)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize/32)*32)

            images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            lateral_map = model(images)
            lateral_map = F.softmax(lateral_map,dim=1)
            for i in range(lateral_map.shape[0]):
                values, indices = lateral_map[i,:].topk(1) #(0.3,0.2,0.5)->(0.3,0.5) (0,2)
                logit_result.append(np.array(indices.cpu(),dtype=np.int16))
            

            logit_data.append(lateral_map.cpu())
            label_list.extend(label.cpu())
            

        logit_data =  torch.cat(logit_data,dim=0).cpu().numpy()
        logit_data = logit_data
        print(logit_data.shape,len(logit_result), len(label_list))

        accuracy = accuracy_score(label_list, logit_result)
        print(accuracy)

        auroc = roc_auc_score(label_list, logit_data, multi_class='ovr')
        print(accuracy)#0.63 
        print(auroc)#0.92


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--train_batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--val_batchsize', type=int,
                        default=128, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='Res2Net')
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = res2net50().to(device)
    model.load_state_dict(torch.load('snapshots\Res2Net\ResNet-3.pth'))
    
    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = 'img'
    csv_file_train = 'train.csv'
    csv_file_val = 'val.csv'
    csv_file_test = 'test.csv'

    with open("class_names.txt","r")  as file:
        clothing_items = file.readlines()
        clothing_items = [cloth.split("\n")[0] for cloth in clothing_items]

    train_loader = get_loader(image_root,csv_file_train, 'train', clothing_items, batchsize=opt.train_batchsize, trainsize=opt.trainsize, shuffle=True)
    val_loader = get_loader(image_root,csv_file_val, 'val', clothing_items, batchsize=opt.val_batchsize, trainsize=opt.trainsize, shuffle=False)
    # test_loader = get_loader(image_root,csv_file_test, 'test', batchsize=opt.val_batchsize, trainsize=opt.trainsize, shuffle=False)

    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    print(total_step)
    
    adjust_lr(optimizer, opt.lr, opt.decay_rate, opt.decay_epoch)
    train(train_loader, val_loader, model, optimizer)
        #gc.collect()
        #torch.cuda.empty_cache()

