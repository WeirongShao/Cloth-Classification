import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, csv_train, tran_or_val, clothing_items, trainsize=224):
        self.trainsize = trainsize
        self.csv_train = csv_train
        self.clothing_items = clothing_items
        self.images = []
        self.images1 = []
        print(self.csv_train)
        self.encoder_class(self.csv_train)

        for f in os.listdir(image_root):
            if f.split('_')[0]== tran_or_val:
                label_train = self.dict_train[f]
                if f.endswith('.jpg') or f.endswith('.png'):
                    self.images.append([os.path.join(image_root,f), label_train])


        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]) 

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index][0])
        image = self.img_transform(image) #PIL:(242,242,3) -> tensor:(3,224,224)
        label = self.images[index][1]
        #tensor:(3,224,224), 3
        return image,label

    # def filter_files(self):
    #     self.image = self.images + self.images1 + self.images2
    #     return self.image

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def resize(self, img, gt):
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

    def encoder_class(self,csv_train):
 
        data_X = pd.read_csv(csv_train) 
        x = data_X[['img_name', 'class_name']]
        x = np.array(x)
        y = [self.clothing_items.index(i) for i in x[:,1]]
        dict_train = dict(zip(x[:,0],y))
        self.dict_train = dict_train



def get_loader(image_root, csv_file, tran_or_val, clothing_items,batchsize, trainsize, shuffle=True, num_workers=2, pin_memory=True):


    if tran_or_val=="train" or tran_or_val=="val":
        dataset = PolypDataset(image_root,csv_file, tran_or_val, clothing_items, trainsize)
    else:
        dataset = Polyp_test_Dataset(image_root,csv_file, tran_or_val, clothing_items, trainsize)
    

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class Polyp_test_Dataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, csv_train, tran_or_val, clothing_items, trainsize=224):
        self.trainsize = trainsize
        self.csv_train = csv_train
        self.clothing_items = clothing_items
        self.images = []
        self.images1 = []
        print(self.csv_train)
        # self.encoder_class(self.csv_train)


        self.images = os.listdir(image_root)
        self.images = [[os.path.join(image_root,image), image.split('_')[1].split('.')[0]] for image in self.images if image.split('_')[0]==tran_or_val]
        
        self.images = sorted(self.images,key=self.funcsort)

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    def funcsort(self,images):
        idx = int(images[1])
        return idx
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index][0])
        image = self.img_transform(image)
        idx = self.images[index][1]
        return image,idx

    # def filter_files(self):
    #     self.image = self.images + self.images1 + self.images2
    #     return self.image



    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def resize(self, img, gt):
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

    def encoder_class(self,csv_train):
 
        data_X = pd.read_csv(csv_train) 
        x = data_X[['idx', 'img_name']]
        self.test_list = np.array(x)


if __name__=="__main__":
    # path = 'train.csv'
    # data1 = pd.read_csv(path) 
    # x = data1[['img_name', 'class_name']]
    # x = np.array(x)
    # encoder1 = LabelEncoder()
    # encoder1.fit(x[:,1])
    # y = encoder1.transform(x[:,1])
    # dict_train = dict(zip(x[:,0],y))

    # print(dict_train)
    # dataset = PolypDataset('img','train.csv')
    image_root = 'img'
    csv_file_train = 'train.csv'
    csv_file_val = 'val.csv'
    csv_file_test = 'test.csv'

    with open("class_names.txt","r")  as file:
        clothing_items = file.readlines()
        clothing_items = [cloth.split("\n")[0] for cloth in clothing_items]
    #loader -> data.loader(loader)
    train_loader = get_loader(image_root,csv_file_test, 'test', clothing_items, batchsize=4, trainsize=224, shuffle=True)
    for image,index in train_loader:
        #(b,3,224,224),(b)
        print(image.shape)
        print(index)

