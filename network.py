import torch
import numpy as np
import torch.nn as nn
from config import Parameters

class point_pillar_net(nn.Module, params:Parameters):

    def __init__(self,in_channels, out_channels):
        super.__init__()
        self.name = 'Pillar Feature Net'
        self.conv1 = nn.Conv2d(in_channels,64,kernel_size=1,stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(64)
        
        
    def forward(self, x): # batch * 12000 * 100 * 4
        pillars_center = pillar_center_get(x) # batch * 12000 * 3, pillar的中心坐标

        x = pillar_form(x) # => batch * 12000 * 100 * 9
        x = self.conv1(x) # => batch * 12000 * 100 * 64
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = pillar_feature_extract(x) # => batch * 12000 * 1 * 64
        x = feature_reshape(x) # => batch * 12000 * 64

         
        x = generate_image(x, pillars_center) # => batch * 504 * 504 * 64
        
        
class pillar_feature_net(nn.Module):
    """
    Extract Each Pillar's feature
    """
    def __init__(self,
                 num_input_features=4, # 4 dim -> 9 dim
                 voxel_size = (cfg.x_step, cfg.y_step)
                 pc_range = (cfg.x_min,cfg.x_max,cfg.y_min,cfg.y_max,cfg.z_min,cfg.z_max)):
       
        super().__init__()
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
    
    def forward(self, features, num_voxels, coors):
        