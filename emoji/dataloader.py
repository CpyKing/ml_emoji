import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import numpy as np
from utils import one_hot_encode

class CustomImageDataset(Dataset):
    def __init__(self, data_path='train_scale', mode='train', height=120, width=120, transformations=None):
        self.height = height
        self.width = width
        self.data_path = data_path
        self.mode = mode
        if self.mode == 'train':
            self.data2labels = pd.read_csv(os.path.join('./data/EMOJI', self.data_path + '_labels', 'labels.csv'))
        else:
            # test
            self.test_data_list = list()
            for img_name in os.listdir('./data/EMOJI/test_imgs'):
                self.test_data_list.append(img_name)
        if transformations is None:
            transformations = transforms.Compose([transforms.Resize((self.width, self.height)),
                                                    transforms.ToTensor()])
        self.transformations = transformations

    def __len__(self):
        if self.mode == 'train':
            return len(self.data2labels)
        else:
            return len(self.test_data_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            single_img_label = self.data2labels.loc[idx, 'cls_label']
            single_img_label = one_hot_encode(single_img_label)
            single_img_name = self.data2labels.loc[idx, 'img_name']
            single_img = Image.open(os.path.join('./data/EMOJI', self.data_path + '_imgs', single_img_name)).convert('RGB')
            single_img = self.transformations(single_img)
        else:
            single_img = Image.open(os.path.join('./data/EMOJI/test_imgs', self.test_data_list[idx])).convert('RGB')
            single_img = self.transformations(single_img)
            single_img_label = self.test_data_list[idx]
        return single_img, single_img_label




if __name__ == '__main__':
    cd = CustomImageDataset()
    # for i in range(len(cd)):
    #     if cd.__getitem__(i)[0].shape[1] != 120:
    #         print(cd.__getitem__(i)[0].shape)clear
    train_dataloader = DataLoader(cd, batch_size=50, shuffle=True)
    print(next(iter(train_dataloader))[1])
