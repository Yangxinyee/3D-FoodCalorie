import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

from torch.autograd import Variable
import pdb
import numpy as np
from necks import BFP

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(planes * 4)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #
        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


rgbd = False  # Change this when running RGBD experiments. lmj 0805-----Deprecated, because 4-channel training is not good.


class ResNet(nn.Module):

    def __init__(self, block, layers, rgbd, bbox, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # if not rgbd:
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, #3->4   padding=3->4
        #                     bias=False)
        # elif rgbd: # Discarded
        #     self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=7, stride=2, padding=4, #3->4   padding=3->4
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # lmj
        # self.fc1 = nn.Linear(2048, 2048)  # 2048
        # self.fc2 = nn.Linear(2048, 2048)
        # self.dropout = nn.Dropout(0.5)
        # self.calorie = nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 1))
        # self.mass = nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 1))
        # self.fat = nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 1))
        # self.carb = nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 1))
        # self.protein = nn.Sequential(nn.Linear(2048, 1024), nn.Linear(1024, 1))

        # FPN lmj 20210831
        # Top layer
        # for conv5
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        # for conv4,conv3,conv2
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        # for conv4,conv3,conv2
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.rgbd = rgbd
        self.yolobox = bbox
        self.adaAvgPool = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool_rgbonly = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(1024, 1024)

    # FPN
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # rgb-only+FPN
    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #     # pdb.set_trace()
    #     c1 = F.relu(self.bn1(self.conv1(x)))
    #     c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
    #     c2 = self.layer1(c1)
    #     c3 = self.layer2(c2)
    #     c4 = self.layer3(c3)
    #     c5 = self.layer4(c4)
    #     # Top-down
    #     p5 = self.toplayer(c5)
    #     p4 = self._upsample_add(p5, self.latlayer1(c4))
    #     p3 = self._upsample_add(p4, self.latlayer2(c3))
    #     p2 = self._upsample_add(p3, self.latlayer3(c2))
    #     # Smooth
    #     p4 = self.smooth1(p4) #原图：（267，356）->(17,23)
    #     p3 = self.smooth2(p3) #原图：（267，356）->(34,45)
    #     p2 = self.smooth3(p2) #原图：（267，356）->(67,89)

    #     cat0 = self.avgpool_rgbonly(p2)
    #     cat1 = self.avgpool_rgbonly(p3)
    #     cat2 = self.avgpool_rgbonly(p4)
    #     cat3 = self.avgpool_rgbonly(p5)

    #     cat_input = torch.stack([cat0, cat1, cat2, cat3], axis=1) # torch.Size([16, 4, 512, 1, 1])
    #     input = cat_input.view(cat_input.shape[0], -1)

    #     x = self.fc3(input)
    #     x = F.relu(x)
    #     results = []
    #     results.append(self.calorie(x).squeeze())
    #     results.append(self.mass(x).squeeze())
    #     results.append(self.fat(x).squeeze())
    #     results.append(self.carb(x).squeeze())
    #     results.append(self.protein(x).squeeze())
    #     return results

    # rgb-only+yolobbox
    def _forward_impl_bbox(self, x, bbox):
        # Bottom-up  FPN
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)  # torch.Size([1, 2048, 8, 8]) when image input ==(256,256)
        # pdb.set_trace()
        # c5 = self.adaAvgPool(c5)

        # before 1108
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)  # [b, 256, 67, 89]
        # (267, 356) ->p2(67,89)
        # pdb.set_trace()

        # H,W = 67,89
        # p2 = self.smooth1(F.upsample(self.toplayer(c5), size=(H,W), mode='bilinear'))

        output = []
        for i, box in enumerate(bbox):
            if box != '':
                # pdb.set_trace()
                with open(box, "r+", encoding="utf-8", errors="ignore") as f:
                    # w,h = 89, 67   # after resize
                    w, h = p2.shape[3], p2.shape[2]
                    allLabels = []
                    for line in f:
                        label = []
                        aa = line.split(" ")
                        # pdb.set_trace()
                        x_center = w * float(aa[1])  # aa[1]upper left x
                        y_center = h * float(aa[2])  # aa[2]upper left y
                        width = int(w * float(aa[3]))  # aa[3] width
                        height = int(h * float(aa[4]))  # aa[4] height
                        lefttopx = int(x_center - width / 2.0)
                        lefttopy = int(y_center - height / 2.0)
                        label = [lefttopx, lefttopy, lefttopx + width, lefttopy + height]
                        allLabels.append(label)

                    nparray = np.array(allLabels)
                    # nultiple labels posistions
                    lefttopx = nparray[:, 0].min()
                    lefttopy = nparray[:, 1].min()
                    # width = nparray[:,2].max()
                    # height = nparray[:,3].max()
                    left_plus_width = nparray[:, 2].max()
                    top_plus_height = nparray[:, 3].max()

                    # pdb.set_trace()
                    roi = p2[i][..., lefttopy + 1:top_plus_height + 3, lefttopx + 1:left_plus_width + 1]
                    # pooling to the same size
                    output.append(F.adaptive_avg_pool2d(roi, (2, 2)))
            elif box == '':
                # pdb.set_trace()
                output.append(F.adaptive_avg_pool2d(p2[i], (2, 2)))
        output = torch.stack(output, axis=0)
        x = torch.flatten(output, 1)
        x = self.fc3(x)
        x = F.relu(x)
        results = []
        results.append(self.calorie(x).squeeze())  # 2048
        results.append(self.mass(x).squeeze())
        results.append(self.fat(x).squeeze())
        results.append(self.carb(x).squeeze())
        results.append(self.protein(x).squeeze())
        return results

    # Normal
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if not self.rgbd:
            # pdb.set_trace()
            # torch.Size([32, 3, 256, 256])#->torch.Size([32, 64, 128, 128])
            x = self.conv1(x)  # torch.Size([16, 3, 267, 356])
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # torch.Size([16, 2048, 9, 12])

            # pdb.set_trace()
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # x = self.fc(x)
            x = self.fc1(x)
            # 0721
            # x = self.dropout(x)
            x = self.fc2(x)
            # 0722
            # x = self.dropout(x)
            # pdb.set_trace()
            x = F.relu(x)
            results = []
            results.append(self.calorie(x).squeeze())  # 2048
            results.append(self.mass(x).squeeze())
            results.append(self.fat(x).squeeze())
            results.append(self.carb(x).squeeze())
            results.append(self.protein(x).squeeze())
            return results

        elif self.rgbd:
            # Bottom-up  FPN
            c1 = F.relu(self.bn1(self.conv1(x)))
            c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)  # torch.Size([1, 2048, 8, 8]) when image input ==(256,256)
            # pdb.set_trace()
            # c5 = self.adaAvgPool(c5)
            # Top-down
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p2 = self._upsample_add(p3, self.latlayer3(c2))
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)
            return p2, p3, p4, p5

    # 20211102
    def forward(self, x, bbox=None):
        # 20211102
        if self.yolobox:
            return self._forward_impl_bbox(x, bbox)
        else:
            return self._forward_impl(x)


# lmj 20210831
class Resnet101_concat(nn.Module):
    def __init__(self):
        super(Resnet101_concat, self).__init__()
        # self.rgb_tensor = rgb
        # self.rgbd_tensor = rgbd
        # pdb.set_trace()
        self.refine = BFP(512, 4)

        self.smooth1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.ca0 = ChannelAttention(512)
        self.sa0 = SpatialAttention()
        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(512)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(512)
        self.sa3 = SpatialAttention()

        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_4 = nn.AdaptiveAvgPool2d((1, 1))

        # self.calorie = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,1))
        # self.mass = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,1))
        # self.fat = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,1))
        # self.carb = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,1))
        # self.protein = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,1))
        # self.fc = nn.Linear(2048, 2048)
        self.calorie = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.mass = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.fat = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.carb = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.protein = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.fc = nn.Linear(2048, 1024)
        self.LayerNorm = nn.LayerNorm(2048)

    # 4 vectors, one result
    def forward(self, rgb, rgbd):
        # pdb.set_trace()
        cat0 = torch.cat((rgb[0], rgbd[0]), 1)  # torch.Size([16, 512, 64, 64])
        cat1 = torch.cat((rgb[1], rgbd[1]), 1)  # torch.Size([16, 512, 32, 32])
        cat2 = torch.cat((rgb[2], rgbd[2]), 1)  # torch.Size([16, 512, 16, 16])
        cat3 = torch.cat((rgb[3], rgbd[3]), 1)  # torch.Size([16, 512, 8, 8])
        # BFP
        cat0, cat1, cat2, cat3 = self.refine(tuple((cat0, cat1, cat2, cat3)))
        # two-modality fusion
        cat0 = self.smooth1(cat0)  # torch.Size([16, 512, 64, 64])
        cat1 = self.smooth1(cat1)  # torch.Size([16, 512, 32, 32])
        cat2 = self.smooth1(cat2)  # torch.Size([16, 512, 16, 16])
        cat3 = self.smooth1(cat3)  # torch.Size([16, 512, 8, 8])
        # CMBA
        cat0 = self.ca0(cat0) * cat0
        cat0 = self.sa0(cat0) * cat0
        cat1 = self.ca1(cat1) * cat1
        cat1 = self.sa1(cat1) * cat1
        cat2 = self.ca2(cat2) * cat2
        cat2 = self.sa2(cat2) * cat2
        cat3 = self.ca3(cat3) * cat3
        cat3 = self.sa3(cat3) * cat3

        cat0 = self.avgpool_1(cat0)
        cat1 = self.avgpool_2(cat1)
        cat2 = self.avgpool_3(cat2)
        cat3 = self.avgpool_4(cat3)

        # pdb.set_trace()

        cat_input = torch.stack([cat0, cat1, cat2, cat3], axis=1)  # torch.Size([16, 4, 512, 1, 1])
        input = cat_input.view(cat_input.shape[0], -1)  # torch.Size([N, 5, 1024]) N =16(bz) 11
        # pdb.set_trace()
        # input = self.LayerNorm(input)
        input = self.fc(input)
        input = F.relu(input)  # torch.Size([16, 2048]) same as faster rcnn

        results = []
        results.append(self.calorie(input).squeeze())
        results.append(self.mass(input).squeeze())
        results.append(self.fat(input).squeeze())
        results.append(self.carb(input).squeeze())
        results.append(self.protein(input).squeeze())

        return results

    # 4 vectors, one result
    # def forward(self, rgb, rgbd):
    #     cat0 = torch.cat((rgb[0],rgbd[0]), 1) #torch.Size([16, 512, 64, 64])
    #     cat1 = torch.cat((rgb[1],rgbd[1]), 1) #torch.Size([16, 512, 32, 32])
    #     cat2 = torch.cat((rgb[2],rgbd[2]), 1) #torch.Size([16, 512, 16, 16])
    #     cat3 = torch.cat((rgb[3],rgbd[3]), 1) #torch.Size([16, 512, 8, 8])

    #     # two-modality fusion
    #     cat0 = self.smooth1(cat0) #torch.Size([16, 512, 64, 64])
    #     cat1 = self.smooth1(cat1) #torch.Size([16, 512, 32, 32])
    #     cat2 = self.smooth1(cat2) #torch.Size([16, 512, 16, 16])
    #     cat3 = self.smooth1(cat3) #torch.Size([16, 512, 8, 8])

    #     cat0 = self.avgpool_1(cat0)
    #     cat1 = self.avgpool_2(cat1)
    #     cat2 = self.avgpool_3(cat2)
    #     cat3 = self.avgpool_4(cat3)

    #     # pdb.set_trace()

    #     # cat_input = torch.stack([cat0, cat1, cat2, cat3], axis=1) # torch.Size([16, 4, 512, 1, 1])
    #     cat_input = cat0 + cat1 + cat2 + cat3
    #     input = cat_input.view(cat_input.shape[0], -1) # torch.Size([N, 5, 1024]) N =16(bz) 11
    #     input = self.fc(input)
    #     input = F.relu(input) # torch.Size([16, 2048])

    #     results = []
    #     results.append(self.calorie(input).squeeze())
    #     results.append(self.mass(input).squeeze())
    #     results.append(self.fat(input).squeeze())
    #     results.append(self.carb(input).squeeze())
    #     results.append(self.protein(input).squeeze())

    #     return results


class Resnet101_Ctran_concat(nn.Module):
    def __init__(self, args, layers=3, heads=4, dropout=0.1):
        super(Resnet101_Ctran_concat, self).__init__()
        # self.rgb_tensor = rgb
        # self.rgbd_tensor = rgbd
        # pdb.set_trace()
        self.batch_size = args.b

        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        # self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool_4 = nn.AdaptiveAvgPool2d((1, 1))

        self.calorie = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 1))
        self.mass = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 1))
        self.fat = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 1))
        self.carb = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 1))
        self.protein = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 1))
        self.fc = nn.Linear(512, 1024)
        ###################################################################################################
        hidden = 1024  # this should match the backbone output feature size
        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Output
        self.output_linear = torch.nn.Linear(hidden, 5)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone

        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    # rgb and rgbd fusion by self-attention
    def forward(self, rgb, rgbd):
        # pdb.set_trace()
        p2, p3, p4, p5 = self.avgpool(rgb[0]), self.avgpool(rgb[1]), self.avgpool(rgb[2]), self.avgpool(
            rgb[3])  # 64; 32; 16; 8->16
        d2, d3, d4, d5 = self.avgpool(rgbd[0]), self.avgpool(rgbd[1]), self.avgpool(rgbd[2]), self.avgpool(
            rgbd[3])  # 64; 32; 16; 8-16

        rgb_cat = torch.cat((p2, p3, p4, p5), 1)  # torch.Size([16, 1024, 16, 16])
        rgbd_cat = torch.cat((d2, d3, d4, d5), 1)
        rgb_cat = rgb_cat.view(rgb_cat.size(0), rgb_cat.size(1), -1).permute(0, 2, 1)  # [16,256,1024]
        rgbd_cat = rgbd_cat.view(rgbd_cat.size(0), rgbd_cat.size(1), -1).permute(0, 2, 1)
        # Concat rgb and depth embeddings
        embeddings = torch.cat((rgb_cat, rgbd_cat), 1)  # torch.Size([16, 512, 1024])
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # pdb.set_trace()
        output = self.output_linear(
            embeddings)  # embeddings.shape = torch.Size([16, 512, 1024]);output.shape = torch.Size([16, 512, 5])
        results = []
        # results.append(self.calorie(F.relu(self.fc(output[:,:,0]))).squeeze())
        # results.append(self.mass(F.relu(self.fc(output[:,:,1]))).squeeze())
        # results.append(self.fat(F.relu(self.fc(output[:,:,2]))).squeeze())
        # results.append(self.carb(F.relu(self.fc(output[:,:,3]))).squeeze())
        # results.append(self.protein(F.relu(self.fc(output[:,:,4]))).squeeze())
        results.append(self.calorie(output[:, :, 0]).squeeze())
        results.append(self.mass(output[:, :, 1]).squeeze())
        results.append(self.fat(output[:, :, 2]).squeeze())
        results.append(self.carb(output[:, :, 3]).squeeze())
        results.append(self.protein(output[:, :, 4]).squeeze())

        return results


def _resnet(arch, block, layers, pretrained, progress, rgbd, bbox, **kwargs):
    model = ResNet(block, layers, rgbd, bbox, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, rgbd=False, bbox=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)


def resnet101(pretrained=False, progress=True, rgbd=False, bbox=False, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


###################################################################################################
'''self-attention'''


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model * 1, dropout=dropout,
                                                         activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu')

    def forward(self, k, mask=None):
        attn = None
        k = k.transpose(0, 1)
        x, attn = self.transformer_layer(k, src_mask=mask)
        # x = self.transformer_layer(k,src_mask=mask)
        x = x.transpose(0, 1)
        return x, attn


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


import math


def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


#####################################################################################################

if __name__ == '__main__':
    # model = torchvision.models.resnet50()
    from PIL import Image
    from torchvision import transforms

    model = resnet101(rgbd=True)
    # print(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # pdb.set_trace()
    img = Image.open("/icislab/volume1/swj/nutrition/nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/dish_1556575014/rgb.png")
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    # input = torch.unsqueeze(img_tensor, dim=0)
    input = torch.randn(1, 3, 256, 256)
    input = input.to(device)
    model.to(device)
    model_cat = Resnet101_concat()
    model_cat.to(device)
    pretrained_dict = torch.load("/icislab/volume1/swj/nutrition/CHECKPOINTS/food2k_resnet101_0.0001.pth")
    now_state_dict = model.state_dict()
    now_state_dict.update(pretrained_dict)
    missing_keys, unexpected_keys = model.load_state_dict(now_state_dict, strict=False)
    # input = torch.randn(1, 3, 256, 256)
    out = model(input)
    out_d = model(input)
    results = model_cat(out, out_d)
    print('debug___________________')