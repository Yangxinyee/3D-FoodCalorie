from __future__ import absolute_import, division, print_function
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
from depth_decoder import DepthDecoder
from depth_encoder import DepthEncoder


class Depth_Model(nn.Module):
    def __init__(self, depth_scale, num_layers=50):
        super(Depth_Model, self).__init__()
        self.depth_scale = depth_scale
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(depth_scale))

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        disp_list = []
        for i in range(self.depth_scale):
            disp = outputs['disp', i]
            disp_list.append(disp)
        return disp_list
