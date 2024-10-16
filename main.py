
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def structure_loss(pred_y, target):
    # print(pred_y,target)
    loss1 = F.cross_entropy(pred_y, target.long())#softmax+nll
    return loss1


def train(train_loader, val_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    loss_record = AvgMeter()
    for i, (packimage,packlabel) in enumerate(train_loader):
    #for packimage,packlabel in train_loader:
        optimizer.zero_grad()
        # ---- data prepare ----
        images = packimage
        label = packlabel
        #print(len(images))
        #[3,5,19,24]
        images = Variable(images).to(device)
        label = Variable(label).to(device)
        # ---- rescale ----
        # trainsize = int(round(opt.trainsize/32)*32)
        # images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

        # ---- forward ----
        #model.cuda()
        lateral_map = model(images)
        # ---- loss function ----
        loss = structure_loss(lateral_map,label)

        # ---- backward ----
        loss.backward()
        # clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----

        loss_record.update(loss.data) #loggor, tensorboard, print


        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))

    #step = 13077/10
    #
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), save_path + 'ResNet-%d.pth' % epoch)
    print('[Saving Snapshot:]', save_path + 'ResNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--train_batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--val_batchsize', type=int,
                        default=1, help='training batch size')
    
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
    model.load_state_dict(torch.load('snapshots\Res2Net\ResNet-1.pth'))
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
        clothing_items = file.readlines()#['a\n','b\n']
        clothing_items = [cloth.split("\n")[0] for cloth in clothing_items]

    train_loader = get_loader(image_root,csv_file_train, 'train', clothing_items, batchsize=opt.train_batchsize, trainsize=opt.trainsize, shuffle=True)
    val_loader = get_loader(image_root,csv_file_val, 'val', clothing_items, batchsize=opt.val_batchsize, trainsize=opt.trainsize, shuffle=False)
    # test_loader = get_loader(image_root,csv_file_test, 'test', batchsize=opt.val_batchsize, trainsize=opt.trainsize, shuffle=False)

    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    print(total_step)
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, val_loader, model, optimizer, epoch)
        #gc.collect()
        #torch.cuda.empty_cache()

