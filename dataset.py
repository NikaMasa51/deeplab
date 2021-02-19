import os

import torch
import torch.nn as nn
from torchvision import  transforms

import numpy as np
from PIL import Image

from data_augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, ToTensor, Normalize_Tensor


class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。
    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                # Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                ToTensor()
                # Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'validation': Compose([
                Resize(input_size),  # リサイズ(input_size)
                ToTensor()
                # Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'validation'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


class DataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, size=1500, transforms=transforms.ToTensor(), data_type="train"):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.data_type = data_type
        self.img_list = os.listdir(f'{img_dir}/{data_type}')
        self.mask_list = os.listdir(f'{mask_dir}/{data_type}')
        self.transforms = transforms
        
    def readImage(self, img_id):
        
        img = Image.open(img_id)
        transform = transforms.Resize((self.size, self.size))
        return transform(img)
    
    def __len__(self) -> int:
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img_id = self.img_list[index]
        
        img = self.readImage(f'{self.img_dir}/{self.data_type}/{img_id}')
        # print(f'{self.img_dir}/{self.data_type}/{img_id}')
        mask = self.readImage(f'{self.mask_dir}/{self.data_type}/{img_id}').convert("L")
        if self.transforms:
            img, mask = self.pull_item(index)
            # img = self.transforms(img)
            # mask = self.transforms(mask)
        sample = {"image": img, "mask": mask}
                 
        return sample 

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(f'{self.img_dir}/{self.data_type}/{image_file_path}')   # [高さ][幅][色RGB]
        # print('img', img)

        # 2. アノテーション画像読み込み
        anno_file_path = self.mask_list[index]
        anno_class_img = Image.open(f'{self.mask_dir}/{self.data_type}/{anno_file_path}')  # [高さ][幅]
        # print('anno_class_img', anno_class_img)

        # 3. 前処理を実施
        img, anno_class_img = self.transforms(self.data_type, img, anno_class_img)

        return img, anno_class_img
        
        
class Prediction(torch.utils.data.Dataset):
    def __init__(self, img_dir, size=256, transforms=transforms.ToTensor()):
        super().__init__()
        self.img_dir = img_dir
        self.size = size
        self.img_list = os.listdir(f'{img_dir}')
        self.transforms = transforms
        
    def readImage(self, img_id):
        img = Image.open(img_id)
        transform = transforms.Resize((self.size, self.size))
        return transform(img)
    
    def __len__(self) -> int:
        
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img_id = self.img_list[index]
        
        img = self.readImage(f'{self.img_dir}/{img_id}')

        if self.transforms:
            img = self.transforms(img)
                 
        return img, img_id
