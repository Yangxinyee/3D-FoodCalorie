# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
'''https://blog.csdn.net/u013841196/article/details/82941410 采用网络多阶段特征融合'''
'''Use multi-stage feature fusion in the network, see: https://blog.csdn.net/u013841196/article/details/82941410'''
'''比train.py多了portion_independent和Direct Prediction的判断'''
'''Compared to train.py, this file adds checks for portion_independent and Direct Prediction'''
from re import S
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.models.inception import InceptionOutputs
import torchvision.transforms as transforms

import os
import argparse

from models import *
from models import myresnet
# from models.myresnet import resnet101,resnet50
from timm.models import *
from timm.models import create_model
from utils.utils import progress_bar,load_for_transfer_learning,logtxt,check_dirs
from utils_data import get_DataLoader
from utils.utils_scheduler import WarmupCosineSchedule
from models.model_from_retrievalnet import T2TNutrition, ViTNutrition


from mydataset import Food
# from torch.utils.tensorboard import SummaryWriter
from models.InceptionV3 import Inception3, Inception3_concat #导入模型更换为InceptionV3_fusion'''
# from models.model_from_retrievalnet import 
import pdb
# from torchsummary import summary
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
import csv
import random
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
import statistics
# from ptflops import get_model_complexity_info


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.9, type=float, help='weight decay') # 5e-4
parser.add_argument('--min_lr', default=2e-4, type=float, help='minimal learning rate')#2e-4
parser.add_argument('--dataset', choices=["nutrition_rgbd","nutrition_rgb","food101","food172","cub200/CUB_200_2011","cifar10","cifar100"], default='cifar10',
                    help='cifar10 or cifar100')
parser.add_argument('--b', type=int, default=8,
                    help='batch size')
parser.add_argument('--resume', '-r', type=str,
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num_classes', type=int, default=1024, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='T2t_vit_t_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"：必须和t2t_vit.py中的 default_cfgs 命名相同')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop_connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn_tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn_momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn_eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
# Transfer learning
parser.add_argument('--transfer_learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer_model', type=str, default=None,
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer_ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')

parser.add_argument('--data_root', type=str, default = "/icislab/volume1/swj/nutrition/nutrition5k/nutrition5k_dataset", help="our dataset root")
parser.add_argument('--run_name',type=str, default="editname")
parser.add_argument('--print_freq', type=int, default=200,help="the frequency of write to logtxt" )
parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
# parser.add_argument('--cls_num', default=101, type=int, metavar='N', help='class number') #172 #见上面 num_classes
parser.add_argument('--mul_cls_num', default=174, type=int, metavar='N', help='ingradient class number') #353 
parser.add_argument('--multi_task',action='store_true',  help='multi-task classification')
parser.add_argument('--pool', default='spoc', type=str, help='pool function')
parser.add_argument('--embed_dim', default=384, type=int, help='T2t_vit_7,T2t_vit_10,T2t_vit_12:256;\
T2t_vit_14:384; T2t_vit_19:448; T2t_vit_24:512')
parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
parser.add_argument('--portion_independent',action='store_true',  help='Nutrition5K: Portion Independent Model')
parser.add_argument('--direct_prediction',action='store_true',  help='Nutrition5K: direct_prediction Model')
parser.add_argument('--rgbd',action='store_true',  help='4 channels')
parser.add_argument('--gradnorm',action='store_true',  help='GradNorm')
parser.add_argument('--alpha', '-a', type=float, default=0.12)
parser.add_argument('--sigma', '-s', type=float, default=100.0)
parser.add_argument('--rgbd_zscore',action='store_true',  help='4 channels')#train+test标准化
parser.add_argument('--rgbd_zscore_foronly_train_or_test_respectedly',action='store_true',  help='4 channels') #分别对train标准化和对test标准化
parser.add_argument('--rgbd_minmax',action='store_true',  help='4 channels')
parser.add_argument('--rgbd_after_check', action='store_true',  help='remained data after we check the dataset')
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--mixup',action='store_true',  help='data augmentation')
parser.add_argument('--use_detect_label',action='store_true',  help='data augmentation')
parser.add_argument('--use_detect_label_cutfeaturemap',action='store_true',  help='需要把transforms.CenterCrop((256,256))去除')

args = parser.parse_args()

set_seed(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
global_step = 0
#lmj global_step as a global variable
# writer = SummaryWriter(log_dir=os.path.join("/icislab/volume1/swj/nutrition/runs/news", f'checkpoint_{args.dataset}_{args.model}_{args.run_name}'))

# Data
print('==> Preparing data..')


print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
global net
#transformer
# net = create_model(
#     args.model,
#     pretrained=args.pretrained,
#     # num_classes=args.num_classes,
#     drop_rate=args.drop,
#     drop_connect_rate=args.drop_connect, 
#     drop_path_rate=args.drop_path,
#     drop_block_rate=args.drop_block,
#     global_pool=args.gp,
#     bn_tf=args.bn_tf,
#     bn_momentum=args.bn_momentum,
#     bn_eps=args.bn_eps,
#     checkpoint_path=args.initial_checkpoint,
#     img_size=args.img_size)

#resnet50
# net = torchvision.models.resnet50(pretrained=True)
# net.fc = nn.Linear(2048, 101)
# pdb.set_trace()


#vit
# pdb.set_trace()
# net = create_model("vit_base_patch16_224", pretrained = True)
# net.head = nn.Linear(768, 101)



# pdb.set_trace()
if args.model == 'resnet101':
    # if not args.rgbd:
    print('==> Load checkpoint..')
    resnet101_food2k = torch.load("/icislab/volume1/swj/nutrition/CHECKPOINTS/food2k_resnet101_0.0001.pth")
    pretrained_dict = resnet101_food2k
    # pretrained_dict = torchvision.models.resnet101(pretrained=True).state_dict()
    #food101
    # pretrained_dict = torch.load("/home/isia/lmj/20210303/Experiment/backbone/resnet101/food101_resnet101_fordet.pth")
    # pretrained_dict = torch.load("./pretained_models/myFusionNet/rgb_best_pre.pth")
    if args.rgbd:
        net = myresnet.resnet101(rgbd = args.rgbd)
        net2 = myresnet.resnet101(rgbd = args.rgbd)
        # pdb.set_trace()
        # net = create_model("vit_base_r50_s16_384", pretrained=True) #vit_base_r50_s16_384
        # net = create_model("vit_base_r50_s16_256mine", pretrained=True) #vit_base_r50_s16_384
        # net_cat = create_model("vit_base_r50_s16_384", pretrained=True)
        net_cat = myresnet.Resnet101_concat()
        # net_cat = myresnet.Resnet101_Ctran_concat(args)
    elif args.use_detect_label_cutfeaturemap:
        net = myresnet.resnet101(bbox = args.use_detect_label_cutfeaturemap)
    else:
        net = myresnet.resnet101()
elif args.model == 'resnet18':
    pretrained_dict = torchvision.models.resnet18(pretrained=True).state_dict()
    net = myresnet.resnet18()
elif args.model == 'resnet50':
    net = myresnet.resnet50()
elif args.model == 'inceptionv3':
    inceptionv3 = torchvision.models.inception_v3(pretrained=True)
    # features =  dict(list(inceptionv3.named_children())[:-3])
    pretrained_dict = inceptionv3.state_dict()
    net = Inception3(aux_logits=False, transform_input = False, rgbd = args.rgbd)
    if args.rgbd:
        net2 = Inception3(aux_logits=False, transform_input = False, rgbd = args.rgbd)
        net_cat = Inception3_concat(args)

elif 'vit_base' in  args.model:
    '''vit_base_patch16_224_in21k  / vit_base_patch16_224''' 
    # pdb.set_trace()
    meta=dict()
    if args.model == 'vit_base_patch16_224_in21k':
        pretrainedvit = create_model("vit_base_patch16_224_in21k", pretrained = True) 
        net = vit.vit_base_patch16_224_in21k(pretrained=False) # pretrained=False 时往vit.py的init函数加新东西没问题
    elif args.model == 'vit_base_patch16_224':
        pretrainedvit = create_model("vit_base_patch16_224", pretrained = True) 
        net = vit.vit_base_patch16_224(pretrained=False)
    pretrained_dict = pretrainedvit.state_dict()
    in_feature = getattr(net, "head").in_features
    meta['in_feature'] = in_feature
    features = dict(list(net.named_children())[:-1])
    net = ViTNutrition(features, meta)
    
elif 'T2t_vit' in args.model:
    # pdb.set_trace()
    meta = {}
    meta['img_size'] = args.img_size
    meta['embed_dim'] = args.embed_dim
    
    net = create_model(args.model, pretrained=args.pretrained, drop_rate=args.drop, drop_connect_rate=args.drop_connect, drop_path_rate=args.drop_path,drop_block_rate=args.drop_block,global_pool=args.gp,bn_tf=args.bn_tf,
    bn_momentum=args.bn_momentum,bn_eps=args.bn_eps,checkpoint_path=args.initial_checkpoint,img_size=args.img_size)

    if args.transfer_learning:
        print('transfer learning, load t2t-vit pretrained model')
        load_for_transfer_learning(net, args.transfer_model, use_ema=True, strict=False, num_classes=args.num_classes)

    in_feature = getattr(net, "head").in_features # net  net.module       .in_features/out_features
    meta['in_feature'] = in_feature
    features = dict(list(net.named_children())[:-1]) 
    pretrained_dict = net.state_dict()
    net = T2TNutrition(features,meta)

# pdb.set_trace()


# if not args.rgbd : 
    # pretrained_dict = net.state_dict()
model_dict = net.state_dict()
new_state_dict = OrderedDict()
# pdb.set_trace()
for k, v in pretrained_dict.items():
    # pdb.set_trace()
    if k in model_dict: #update the same part
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
# pdb.set_trace()
model_dict.update(new_state_dict)
net.load_state_dict(model_dict)
# pdb.set_trace()
if args.rgbd:
    net2.load_state_dict(model_dict)



# #T2T
# if args.transfer_learning:
#     print('transfer learning, load t2t-vit pretrained model')
#     load_for_transfer_learning(net, args.transfer_model, use_ema=True, strict=False, num_classes=args.num_classes)

# pdb.set_trace()
net = net.to(device)
if args.rgbd:
    net2 = net2.to(device)
    net_cat = net_cat.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

#换位置  293-320


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    if args.rgbd:
        net2 = torch.nn.DataParallel(net2)
        net_cat = torch.nn.DataParallel(net_cat)
    cudnn.benchmark = True

criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
parameters = net.parameters()
# awl = AutomaticWeightedLoss(5)
# optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
# optimizer = optim.SGD(parameters, lr=3e-4, momentum=0.9, weight_decay=5e-4) #1e-5
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)
#Adam + ExponentialLR

# optimizer = torch.optim.Adam([
#     {'params': net.parameters(), 'lr':1e-4, 'weight_decay':1e-5}
#     #  {'params': awl.parameters(), 'weight_decay': 0}
#     ]) 
# pdb.set_trace()
optimizer = torch.optim.Adam([
        {'params': (p for name, p in net.named_parameters() if 'bias' not in name)},
        {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'weight_decay': 0.}
    ], lr=1e-4, weight_decay=5e-4)

if args.rgbd:
    # conv1 = list(map(id, net_cat.module.fat.parameters()))
    # conv2 = list(map(id, net_cat.module.carb.parameters()))
    # conv3 = list(map(id, net_cat.module.protein.parameters()))
    # base_params_net_cat = filter(lambda p: id(p) not in conv1 + conv2 + conv3, net_cat.module.parameters())
    # optimizer = torch.optim.Adam([
    #                             {'params': net.module.parameters()},#5e-4
    #                             {'params': net2.module.parameters()},#5e-4
    #                              {'params': base_params_net_cat},
    #                              {'params': net_cat.module.fat.parameters(),'lr':1e-3, 'weight_decay': 5e-4},
    #                              {'params': net_cat.module.carb.parameters(),'lr':1e-3, 'weight_decay': 5e-4},
    #                              {'params': net_cat.module.protein.parameters(),'lr':1e-3, 'weight_decay': 5e-4},#5e-4
    #                              {'params': awl.parameters(), 'weight_decay': 0}
    #                              ],lr=1e-4, weight_decay= 5e-4) #weight_decay=1e-5  lr=1e-4

    optimizer = torch.optim.Adam([
        {'params': net.module.parameters(),'lr':5e-5, 'weight_decay': 5e-4},#5e-4
        {'params': net2.module.parameters(), 'lr':5e-5, 'weight_decay': 5e-4},#5e-4
         {'params': net_cat.module.parameters(),'lr':5e-5, 'weight_decay': 5e-4}#5e-4
        #  {'params': awl.parameters(), 'weight_decay': 0}
         ]) 
    # optimizer = torch.optim.Adam([
    #     {'params': (p for name, p in net.named_parameters() if 'bias' not in name)},
    #     {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'weight_decay': 0.},
    #     {'params': (p for name, p in net2.named_parameters() if 'bias' not in name)},
    #     {'params': (p for name, p in net2.named_parameters() if 'bias' in name), 'weight_decay': 0.},
    #     {'params': (p for name, p in net_cat.named_parameters() if 'bias' not in name)},
    #     {'params': (p for name, p in net_cat.named_parameters() if 'bias' in name), 'weight_decay': 0.},

    # ], lr=1e-4, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # 0.99 (RGBD )
# optimizer = torch.optim.RMSprop(parameters, lr = args.lr, weight_decay = args.wd, momentum = 0.9, eps=1.0)
# lambda_epoch = lambda e: 1.0 if e < 5 else (5e-1 if e < 20 else 1e-2 if e < 30 else (5e-3 if e < 40 else (1e-3)))
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)
# scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1,t_total=20) 
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

if args.resume:
    # Load checkpoint. resume时不同于上面加载预训练模型，checkpoint网络结构与当前网络一致，因此不需要if k in net.state_dict():
    print('==> Resuming from checkpoint..')
    # pdb.set_trace()
    # check_dir(args.resume)
    # assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./saved/checkpoint/ckpt.pth')
    # checkpoint_path = '/icislab/volume1/swj/nutrition/saved/new/regression_nutrition_rgbd_resnet101_resnet101_food2k_pretrained_fpn_multifusion_w_5scales_bfp_cmba/ckpt_0.0001_0.9_epoch99.pth'
    models_state_dict = torch.load(args.resume)

    new_state_dict_rgb = OrderedDict()
    for k, v in models_state_dict['net'].items():
        name = k[7:] if k.startswith('module') else 'module.' + k
        new_state_dict_rgb[name] = v
    net.load_state_dict(new_state_dict_rgb)

    net2.load_state_dict(models_state_dict['net_d'])
    # print(missing_keys)

    net_cat.load_state_dict(models_state_dict['net_cat'])
    # pdb.set_trace()
    # best_acc = checkpoint['acc']
    #恢复当时的优化器
    optimizer.load_state_dict(models_state_dict['optimizer'])
    #记录resume时继续的epoch编号
    start_epoch = models_state_dict['epoch']
#gradnorm
weights = []
task_losses = []
loss_ratios = []
grad_norm_losses = []

trainloader, testloader = get_DataLoader(args)
# image_sizes = ((256, 352), (320, 448), (384, 512))
image_sizes = ((256, 352), (288, 384), (320, 448), (352, 480), (384, 512))
# Training
def train(epoch,net):
    #lmj global_step作为全局变量
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    if args.rgbd:
        net2.train()
        net_cat.train()
    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    # pdb.set_trace()
    epoch_iterator = tqdm(trainloader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):    #(inputs, targets,ingredient)
        '''Portion Independent Model'''
        #groundtruth:
        # pdb.set_trace()
        # inputs = torch.stack(x[0])

        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        if args.rgbd:
            inputs_rgbd = x[7].to(device)
        if args.use_detect_label_cutfeaturemap:
            bbox = x[9]

        if batch_idx % 10 == 0:
            ns = image_sizes[random.randint(0,4)]
            inputs = F.interpolate(inputs, size=ns, mode='bilinear', align_corners=False)
            inputs_rgbd = F.interpolate(inputs_rgbd, size=ns, mode='bilinear', align_corners=False)

        calories_per = total_calories/total_mass
        fat_per = total_fat/total_mass
        carb_per = total_carb/total_mass
        protein_per = total_protein/total_mass

        #20211018 使用mixup
        # pdb.set_trace()
        if args.mixup:
            if not args.rgbd:
                y = [total_calories, total_mass, total_fat, total_carb, total_protein]
                inputs, y_a, y_b, lam = mixup_data(inputs, y, alpha=0.2)
            else:
                pass

        optimizer.zero_grad() #0817
        # pdb.set_trace()
        #1102
        # pdb.set_trace()
        if args.use_detect_label_cutfeaturemap:
            outputs = net(inputs, bbox)
        else:
            # pdb.set_trace()
            outputs = net(inputs)
        # pdb.set_trace()
        # ops, params = get_model_complexity_info(net, (3, 267, 356), as_strings=True, 
		# 								print_per_layer_stat=True, verbose=True)
        # if args.gradnorm:
            # gradNormModel = RegressionTrain(net,5).to('cuda')
        if args.rgbd:
            if args.model == 'inceptionv3':
                h1,h2,h3,h4,h5 = outputs
                outputs_rgbd = net2(inputs_rgbd)
                # ops, params = get_model_complexity_info(net2, (3, 267, 356), as_strings=True, 
                # 							print_per_layer_stat=True, verbose=True)
                d1,d2,d3,d4,d5 = outputs_rgbd
                outputs = net_cat([h1,h2,h3,h4,h5], [d1,d2,d3,d4,d5])
            elif args.model == 'resnet101':
                p2, p3, p4, p5 = outputs
                outputs_rgbd = net2(inputs_rgbd)
                d2, d3, d4, d5 = outputs_rgbd
                outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])
            # 计算模型大小和运算量
            # Calculate model size and computation
            # ops, params = get_model_complexity_info(net_cat, [h1,h2,h3,h4,h5], [d1,d2,d3,d4,d5], as_strings=True, 
			# 							print_per_layer_stat=True, verbose=True)
            # if args.gradnorm:
            #     gradNormModel = RegressionTrain(net, net2,5).to('cuda')

            # outputs = np.array(outputs) + np.array(outputs_rgbd)
            # outputs = outputs_rgbd
        # pdb.set_trace()
        #loss
        if args.portion_independent:
            # '测试，换一种方法标准化 0728'
            # 'Test, try another normalization method 0728'
            calories_per_loss = criterion(outputs[0], calories_per)
            fat_per_loss = criterion(outputs[2], fat_per)
            carb_per_loss = criterion(outputs[3], carb_per)
            protein_per_loss = criterion(outputs[4], protein_per)
            loss = calories_per_loss + fat_per_loss + carb_per_loss + protein_per_loss

            loss.backward()
            optimizer.step()
            global_step += 1

            train_loss += loss.item()
            calories_loss += calories_per_loss.item()
            mass_loss = 0
            fat_loss += fat_per_loss.item()
            carb_loss += carb_per_loss.item()
            protein_loss += protein_per_loss.item()

        elif args.direct_prediction:
            if args.mixup:
                total_calories_loss =  total_calories.shape[0]* mixup_criterion(criterion, outputs[0], y_a[0], y_b[0], lam) / total_calories.sum().item()
                total_mass_loss =  total_mass.shape[0] * mixup_criterion(criterion, outputs[1], y_a[1], y_b[1], lam)  / total_mass.sum().item()
                total_fat_loss = total_fat.shape[0] *  mixup_criterion(criterion, outputs[2], y_a[2], y_b[2], lam) / total_fat.sum().item()
                total_carb_loss =  total_carb.shape[0] * mixup_criterion(criterion, outputs[3], y_a[3], y_b[3], lam) / total_carb.sum().item()
                total_protein_loss =  total_protein.shape[0] * mixup_criterion(criterion, outputs[4], y_a[4], y_b[4], lam)  / total_protein.sum().item()
            else:
                total_calories_loss = total_calories.shape[0]* criterion(outputs[0], total_calories)  / total_calories.sum().item() 
                total_mass_loss = total_calories.shape[0]* criterion(outputs[1], total_mass)  / total_mass.sum().item()
                total_fat_loss = total_calories.shape[0]* criterion(outputs[2], total_fat)  / total_fat.sum().item()
                total_carb_loss = total_calories.shape[0]* criterion(outputs[3], total_carb) / total_carb.sum().item()
                total_protein_loss = total_calories.shape[0]* criterion(outputs[4], total_protein)  / total_protein.sum().item()

            # total_calories_loss =  criterion(outputs[0], total_calories)  
            # total_mass_loss = criterion(outputs[1], total_mass)  
            # total_fat_loss =  criterion(outputs[2], total_fat) 
            # total_carb_loss =  criterion(outputs[3], total_carb) 
            # total_protein_loss =  criterion(outputs[4], total_protein) 
            
            

            # pdb.set_trace()
            #https://blog.csdn.net/leiduifan6944/article/details/107486857
            # old_loss = torch.tensor([total_calories_loss, total_mass_loss, total_fat_loss,total_carb_loss, total_protein_loss],requires_grad=True)
            # loss_w = get_losses_weights(old_loss)
            # new_loss = old_loss * loss_w
            # loss = new_loss[0] +new_loss[1] + new_loss[2] + new_loss[3] + new_loss[4]


            # loss = awl(total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss)

            #GradNorm
            # if args.gradnorm:
            #     # pdb.set_trace()
            #     task_loss = torch.stack([total_calories_loss, total_mass_loss, total_fat_loss, total_carb_loss, total_protein_loss])
            #     if batch_idx == 0:
            #         # set L(0)
            #         if torch.cuda.is_available():
            #             initial_task_loss = task_loss.data.cpu()
            #         else:
            #             initial_task_loss = task_loss.data
            #         initial_task_loss = initial_task_loss.numpy()
            #
            #     grad_norm_loss = get_grad_norm_losses(gradNormModel,task_loss,initial_task_loss)#net.module,task_loss,initial_task_loss
            #     loss = grad_norm_loss

            loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss
            if not args.gradnorm:
                loss.backward()
            optimizer.step()
            global_step += 1

            train_loss += loss.item()
            calories_loss += total_calories_loss.item()
            mass_loss += total_mass_loss.item()
            fat_loss += total_fat_loss.item()
            carb_loss += total_carb_loss.item()
            protein_loss += total_protein_loss.item()

        '''
        if (batch_idx % 100) == 0:
            # epoch_iterator.set_description(
            #             "Training Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %f" % (epoch, train_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr'])
            #         )
            print("\nTraining Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (epoch, train_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr']))
        '''

        # if args.gradnorm:
        #     # renormalize
        #     n_tasks = 5
        #     normalize_coeff = n_tasks / torch.sum(net.module.weights.data, dim=0)
        #     net.module.weights.data = net.module.weights.data * normalize_coeff
        #
        #     # record
        #     if torch.cuda.is_available():
        #         task_losses.append(task_loss.data.cpu().numpy())
        #         loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
        #         weights.append(net.module.weights.data.cpu().numpy())
        #         grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        #     else:
        #         task_losses.append(task_loss.data.numpy())
        #         loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
        #         weights.append(net.module.weights.data.numpy())
        #         grad_norm_losses.append(grad_norm_loss.data.numpy())
        #
        #     if batch_idx % 20 == 0:
        #         if torch.cuda.is_available():
        #             print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
        #                                 batch_idx, len(trainloader), loss_ratios[-1], net.module.weights.data.cpu().numpy(), task_loss.data.cpu().numpy(), grad_norm_loss.data.cpu().numpy()))
        #         else:
        #             print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
        #                 batch_idx, len(trainloader), loss_ratios[-1], net.module.weights.data.numpy(), task_loss.data.numpy(), grad_norm_loss.data.numpy()))
        
        
        
        #lmj writer to tensorboard
        # writer.add_scalar("train/loss", scalar_value=train_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/calorieloss", scalar_value=calories_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/massloss", scalar_value=mass_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/fatloss", scalar_value=fat_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/carbloss", scalar_value=carb_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/proteinloss", scalar_value=protein_loss/(batch_idx+1), global_step=global_step)
        # writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        

        if (batch_idx+1) % args.print_freq == 0 or batch_idx+1 == len(trainloader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \t'
                    'lr:{:.7f}'.format(
                    epoch, batch_idx+1, len(trainloader), 
                    train_loss/(batch_idx+1), 
                    calories_loss/(batch_idx+1),
                    mass_loss/(batch_idx+1),
                    fat_loss/(batch_idx+1),
                    carb_loss/(batch_idx+1),
                    protein_loss/(batch_idx+1),
                    optimizer.param_groups[0]['lr']))

best_loss = 10000
def test(epoch,net):
    #writer  写入tensorboard
    # global best_acc
    global best_loss
    net.eval()
    if args.rgbd:
        net2.eval()
        net_cat.eval()
    test_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(testloader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    csv_rows = []
    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator): # testloader
            # pdb.set_trace()
            # Portion Independent Model
            #groundtruth:
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()
            if args.rgbd:
                inputs_rgbd = x[7].to(device)
            if args.use_detect_label_cutfeaturemap:
                bbox = x[9]

            # if batch_idx % 10 == 0:
            #     ns = image_sizes[random.randint(0, 2)]
            #     inputs = F.interpolate(inputs, size=ns, mode='bilinear', align_corners=False)
            #     inputs_rgbd = F.interpolate(inputs_rgbd, size=ns, mode='bilinear', align_corners=False)

            calories_per = total_calories/total_mass
            fat_per = total_fat/total_mass
            carb_per = total_carb/total_mass
            protein_per = total_protein/total_mass

            optimizer.zero_grad()
            # outputs = net(inputs)
            if args.use_detect_label_cutfeaturemap:
                outputs = net(inputs, bbox)
            else:
                outputs = net(inputs)
            if args.rgbd:
                if args.model == 'inceptionv3':
                    h1,h2,h3,h4,h5 = outputs
                    outputs_rgbd = net2(inputs_rgbd)
                # ops, params = get_model_complexity_info(net2, (3, 267, 356), as_strings=True, 
                # 							print_per_layer_stat=True, verbose=True)
                    d1,d2,d3,d4,d5 = outputs_rgbd
                    outputs = net_cat([h1,h2,h3,h4,h5], [d1,d2,d3,d4,d5])
                elif args.model == 'resnet101':
                    p2, p3, p4, p5 = outputs
                    outputs_rgbd = net2(inputs_rgbd)
                    d2, d3, d4, d5 = outputs_rgbd
                    outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])
                # outputs = np.array(outputs) + np.array(outputs_rgbd)
                # outputs = outputs_rgbd
            #loss
            if args.portion_independent:
                # '测试，换一种方法标准化 0728'
                # 'Test, try another normalization method 0728'
                calories_total_loss = criterion(outputs[0], calories_per)
                fat_total_loss = criterion(outputs[2], fat_per)
                carb_total_loss = criterion(outputs[3], carb_per)
                protein_total_loss = criterion(outputs[4], protein_per)
                loss = calories_total_loss + fat_total_loss + carb_total_loss + protein_total_loss
                # if batch_idx == len(epoch_iterator)-1:
                #     pdb.set_trace()
                if epoch % 1 ==0:
                    #每10个epoch将当前测试集所有预测值写入csv
                    # Every 10 epochs, write all current test set predictions to csv
                    for i in range(len(x[1])):#IndexError: tuple index out of range  最后一轮的图片数量不到32，不能被batchsiz
                        ############################### 同一类，不同图片的值需要求平均？
                        ############################### For the same class, do different images need to be averaged?
                        dish_id = x[1][i]
                        # pdb.set_trace()
                        calories = outputs[0][i] * total_mass[i]
                        mass =  total_mass[i]
                        fat = outputs[2][i] * total_mass[i]
                        carb = outputs[3][i] * total_mass[i]
                        protein = outputs[4][i] * total_mass[i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)
                    

                # pdb.set_trace()
                test_loss += loss.item()
                calories_loss += calories_total_loss.item()
                mass_loss = 0
                fat_loss += fat_total_loss.item()
                carb_loss += carb_total_loss.item()
                protein_loss += protein_total_loss.item()
            
            elif args.direct_prediction:
                calories_total_loss = total_calories.shape[0]* criterion(outputs[0], total_calories) /total_calories.sum().item()
                mass_total_loss = total_calories.shape[0]* criterion(outputs[1], total_mass)  /total_mass.sum().item()
                fat_total_loss = total_calories.shape[0]* criterion(outputs[2], total_fat) /total_fat.sum().item()
                carb_total_loss = total_calories.shape[0]* criterion(outputs[3], total_carb) /total_carb.sum().item()
                protein_total_loss = total_calories.shape[0]* criterion(outputs[4], total_protein) /total_protein.sum().item()

                # calories_total_loss =  criterion(outputs[0], total_calories) 
                # mass_total_loss =  criterion(outputs[1], total_mass)  
                # fat_total_loss =  criterion(outputs[2], total_fat) 
                # carb_total_loss =  criterion(outputs[3], total_carb) 
                # protein_total_loss =  criterion(outputs[4], total_protein) 


                # pdb.set_trace()
                # old_loss = torch.tensor([calories_total_loss, mass_total_loss, fat_total_loss, carb_total_loss, protein_total_loss])
                # loss_w = get_losses_weights(old_loss)
                # new_loss = old_loss * loss_w
                # loss = new_loss[0] +new_loss[1] + new_loss[2] + new_loss[3] + new_loss[4]


                # loss = awl(calories_total_loss, mass_total_loss, fat_total_loss, carb_total_loss, protein_total_loss)


                loss = calories_total_loss + mass_total_loss+ fat_total_loss + carb_total_loss + protein_total_loss

                # loss =   fat_total_loss + carb_total_loss + protein_total_loss

                if epoch % 1 ==0:
                    #每10个epoch将当前测试集所有预测值写入csv
                    # Every 10 epochs, write all current test set predictions to csv
                    for i in range(len(x[1])):#IndexError: tuple index out of range  最后一轮的图片数量不到32，不能被batchsiz
                        dish_id = x[1][i]
                        calories = outputs[0][i]
                        mass =  outputs[1][i]
                        fat = outputs[2][i]
                        carb = outputs[3][i]
                        protein = outputs[4][i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)

                test_loss += loss.item()
                calories_loss += calories_total_loss.item()
                mass_loss += mass_total_loss.item()
                fat_loss += fat_total_loss.item()
                carb_loss += carb_total_loss.item()
                protein_loss += protein_total_loss.item()


            epoch_iterator.set_description(
                    "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (epoch, test_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr'])
                )
        #lmj writer to tensorboard
        # writer.add_scalar("test/loss", scalar_value=test_loss/(len(testloader)), global_step=global_step)
        # writer.add_scalar("test/calorieloss", scalar_value=calories_loss/(len(testloader)), global_step=global_step) # calories_loss/(batch_idx+1)
        # writer.add_scalar("test/massloss", scalar_value=mass_loss/(len(testloader)), global_step=global_step)
        # writer.add_scalar("test/fatloss", scalar_value=fat_loss/(len(testloader)), global_step=global_step)
        # writer.add_scalar("test/carbloss", scalar_value=carb_loss/(len(testloader)), global_step=global_step)
        # writer.add_scalar("test/proteinloss", scalar_value=protein_loss/(len(testloader)), global_step=global_step)

        # if batch_idx % args.print_freq == 0:
        # pdb.set_trace()
        logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \t'
                    'lr:{:.7f}\n'.format(
                    epoch, batch_idx+1, len(testloader), 
                    test_loss/len(testloader), 
                    calories_loss/len(testloader),
                    mass_loss/len(testloader),
                    fat_loss/len(testloader),
                    carb_loss/len(testloader),
                    protein_loss/len(testloader),
                    optimizer.param_groups[0]['lr']))
    # Save checkpoint.
    # pdb.set_trace()
    if best_loss > test_loss:
        best_loss = test_loss
        print('Saving..')
        net = net.module if hasattr(net, 'module') else net
        state = {
            'net': net.state_dict(),
            'net_d' : net2.state_dict(),
            'net_cat' : net_cat.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch': epoch
        }
        savepath = f"./saved/new/regression_{args.dataset}_{args.model}_{args.run_name}"
        check_dirs(savepath)
        torch.save(state, os.path.join(savepath,f"ckpt_best.pth"))

        #lmj
        # logtxt(log_file_path, "BEST VALID ACCURACY SO FAR: %.2f" % best_acc)
        
    if epoch % 1 == 0:
        new_csv_rows = []
        predict_values = dict()
        # pdb.set_trace()
        key = ''
        for iterator in csv_rows:
            if key != iterator[0]:
                key = iterator[0]
                predict_values[key] = []
                predict_values[key].append(iterator[1:])
            else:
                predict_values[key].append(iterator[1:])
        # pdb.set_trace()
        for k,v in predict_values.items():
            nparray = np.array(v)
            predict_values[k] = np.mean(nparray,axis=0) #每列求均值
            new_csv_rows.append([k, predict_values[k][0], predict_values[k][1], predict_values[k][2], predict_values[k][3], predict_values[k][4]])

        headers = ["dish_id", "calories", "mass", "fat", "carb", "protein"]
        csv_file_path = os.path.join("logs_nutrition2",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',"epoch{}_result_image.csv".format(epoch))
        #每张图片的结果写入csv
        # Write the result of each image to csv
        # with open(csv_file_path,'w')as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow(headers)
        #     f_csv.writerows(csv_rows)
        #每个dish写入csv
        # Write the result of each dish to csv
        csv_file_path2 = os.path.join("/icislab/volume1/swj/nutrition/logs/new",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',"epoch{}_result_dish.csv".format(epoch))
        with open(csv_file_path2,'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(new_csv_rows)

# def grad_norm_losses(losses):
# 	if type(losses) != torch.Tensor:
# 		losses = torch.tensor(losses)
# 	weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
# 	return weights
def get_grad_norm_losses(gradNormModel, task_loss,initial_task_loss):
    # pdb.set_trace()
    weighted_task_loss = torch.mul(gradNormModel.weights, task_loss)
    # if batch_idx == 0:
    #     # set L(0)
    #     if torch.cuda.is_available():
    #         initial_task_loss = task_loss.data.cpu()
    #     else:
    #         initial_task_loss = task_loss.data
    #     initial_task_loss = initial_task_loss.numpy()
    # get the total loss
    loss = torch.sum(weighted_task_loss)
    # clear the gradients
    optimizer.zero_grad()
    # do the backward pass to compute the gradients for the whole set of weights
    # This is equivalent to compute each \nabla_W L_i(t)
    loss.backward(retain_graph=True)

    # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
    #print('Before turning to 0: {}'.format(model.weights.grad))
    gradNormModel.weights.grad.data = gradNormModel.weights.grad.data * 0.0
    #print('Turning to 0: {}'.format(model.weights.grad))

    # get layer of shared weights
    #目前我是将rgb(net)和rgbd(net2)的fc1层concat，如果一开始就concat会怎样？？？？？？？
    if hasattr(net, 'fc2'):
        W = net.get_last_shared_layer()
    else:
        W = net.module.get_last_shared_layer()
    # W2 = net2.get_last_shared_layer()

    # get the gradient norms for each of the tasks
    # G^{(i)}_w(t)
    norms = []
    for i in range(len(task_loss)):
        # get the gradient of this task loss with respect to the shared parameters
        # pdb.set_trace()
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        # compute the norm
        norms.append(torch.norm(torch.mul(gradNormModel.weights[i], gygw[0])))
    norms = torch.stack(norms)
    #print('G_w(t): {}'.format(norms))


    # compute the inverse training rate r_i(t)
    # \curl{L}_i
    # pdb.set_trace()
    if torch.cuda.is_available():
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
    else:
        loss_ratio = task_loss.data.numpy() / initial_task_loss
    # r_i(t)
    inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    #print('r_i(t): {}'.format(inverse_train_rate))


    # compute the mean norm \tilde{G}_w(t)
    if torch.cuda.is_available():
        mean_norm = np.mean(norms.data.cpu().numpy())
    else:
        mean_norm = np.mean(norms.data.numpy())
    #print('tilde G_w(t): {}'.format(mean_norm))

    # pdb.set_trace()
    # compute the GradNorm loss
    # this term has to remain constant
    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
    if torch.cuda.is_available():
        constant_term = constant_term.cuda()
    #print('Constant term: {}'.format(constant_term))
    # this is the GradNorm loss itself
    # grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))
    grad_norm_loss = torch.as_tensor(torch.sum(torch.abs(norms - constant_term))) #https://ask.csdn.net/questions/7413240
    #print('GradNorm loss {}'.format(grad_norm_loss))

    # compute the gradient for the weights
    # pdb.set_trace()
    gradNormModel.weights.grad = torch.autograd.grad(grad_norm_loss, gradNormModel.weights)[0]
    #lmj
    # gradNormModel.weights.grad = torch.autograd.grad(grad_norm_loss, gradNormModel.weights, retain_graph=True)[0]


    return grad_norm_loss
def mixup_data(x, y, alpha=0.2, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    '''x: rgb图像   depth： depth伪彩色图像'''
    '''x: rgb image   depth: depth pseudo-color image'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # pdb.set_trace()
    mixed_x = lam * x + (1 - lam) * x[index,:] # 自己和打乱的自己进行叠加
    mixed_x = lam * x + (1 - lam) * x[index,:] # Mix itself and its shuffled version
    # mixed_depth = lam * depth + (1 - lam) * depth[index,:] #RGB和Depth的index是相同的，lam也相同
    # mixed_depth = lam * depth + (1 - lam) * depth[index,:] # The index and lam for RGB and Depth are the same
    y_a = y
    y_b = []
    for i in range(len(y)):
        y_b.append(y[i][index])
    # pdb.set_trace()

    # y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return  lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# pdb.set_trace()
log_file_path = os.path.join("/icislab/volume1/swj/nutrition/logs/new",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',"train_log.txt")
check_dirs(os.path.join("/icislab/volume1/swj/nutrition/logs/new",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}'))
logtxt(log_file_path, str(vars(args)))
for epoch in range(start_epoch, start_epoch+300):
    train(epoch,net)
    test(epoch,net)
    scheduler.step()




