import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

import imageio
import cv2
import pdb

class Nutrition(Dataset):
    def __init__(self, image_path, txt_dir, transform=None):

        file = open(txt_dir, 'r')
        lines = file.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        # pdb.set_trace()
        for line in lines:
            image = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # Category 1-
            calories = line.strip().split()[2]
            mass =  line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]  # Path for each image
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
       
    def __getitem__(self, index):
        # img = cv2.imread(self.images[index])  
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL
        # except:
        #     print("图片有误：",self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            try:
                # RGB-D images have different sizes, scale according to different ratios
                if 'realsense_overhead' in self.images[index]:
                    # pdb.set_trace()
                    self.transform.transforms[0].size = (267, 356)
                    # print(self.transform)
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('Error in trans_img')
        return img, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[index]

    def __len__(self):
        return len(self.images)


#RGB-D 
class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # Category 1-
            calories = line.strip().split()[2]
            mass =  line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # Path for each image
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)] 

        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform

    #RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])  
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)) # Convert cv2 to PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd,cv2.COLOR_BGR2RGB)) # Convert cv2 to PIL
        except:
            print("Image error:",self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3)) 
        # d_img = np.array(self.my_loader(d_path, 1) ) 
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img)) 
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[index], img_rgbd  # Return 2 types of images, then in train2.py add another judgment, input two images into the network twice

    def __len__(self):
        return len(self.images)
    
    
        
#20210526
class Food(Dataset):

    def __init__(self, txt_dir, image_path, transform=None):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.image_path = image_path
 
    def __len__(self):
        
        return len(self.imgs)
  
    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        # label = list(map(int, label))
        # print label

        # print type(label)
   
        image = Image.open(os.path.join(self.image_path, img_name)).convert('RGB')

        # print img
        if self.transform is not None:
            img = self.transform(image)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label

class ImagesForMulCls(Dataset):

    def __init__(self, args, ims_root, category_list,ingredient_list, imsize=224, bbxs=None, transform=None):

        self.root = ims_root
        if args.dataset == 'food172':
            self.images_fn, self.clusters, self.mul_clusters = self.get_imgs_food172(ims_root, ingredient_list)
        elif args.dataset == 'food101':
            self.images_fn, self.clusters, self.mul_clusters = self.get_imgs_food101(ims_root, category_list, ingredient_list)
        self.imsize = imsize
        self.transform = transform

    def get_imgs_food101(self, ims_root, category_list, ingredient_list):
        # pdb.set_trace()
        if not os.path.exists(category_list) or not os.path.exists(ingredient_list):
            print('!!!THE FILE ROOT NOT EXIST!!!')
            pass
        category_file = open(category_list)
        ingredient_file = open(ingredient_list)
        images = [] # Image paths
        clusters = [] # Category labels
        mul_clusters = [] # Ingredient list

        for line in category_file.readlines():
            image = line.split()[0]  # apple_pie/1005649.jpg
            label = line.strip().split()[1]  # Food category
            images += [os.path.join(ims_root, image)]  # Path for each image
            clusters += [int(label)]  # Corresponding label
        for line in ingredient_file.readlines():
             mult_label = line.strip().split()[1:]  # Ingredient labels
             mul_clusters += [[int(i) for i in mult_label]]  # Ingredient list

        return images, np.array(clusters), np.array(mul_clusters)

    def get_imgs_food172(self, ims_root, ingredient_list):
        pdb.set_trace()
        if not os.path.exists(ingredient_list):
            print(ingredient_list)
            pass
        file = open(ingredient_list)
        lines = file.readlines()
        images = []
        clusters = []
        mul_clusters = []
        for line in lines:
            image = line.split()[0]  # 1/21_20.jpg
            label = line.strip().split()[1]  # Food category
            mult_label = line.strip().split()[2:]  # Ingredient labels

            images += [os.path.join(ims_root, image)]  # Path for each image
            clusters += [int(label)]  # Corresponding label
            mul_clusters += [[int(i) for i in mult_label]]  # Ingredient list
        return images, np.array(clusters), np.array(mul_clusters)

    def __getitem__(self, index):
        img = cv2.imread(self.images_fn[index])  # self.images_fn[index]: image path
        try:
            img = cv2.resize(img, (self.imsize, self.imsize))
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # Convert cv2 to PIL
        except:
            print("Image error:",self.images_fn[index])
            # print("Image error:")
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('Error in trans_img')

        return img, self.clusters[index], self.mul_clusters[index]  # image, category label, ingredient list

    def __len__(self):
        return len(self.images_fn)

class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # pdb.set_trace()
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]
        if not self.is_train:
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]
    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)