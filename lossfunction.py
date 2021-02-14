import torch
from config import Parameters

class point_pillars_loss(nn.Module):

    def __init__(self, params:Parameters):
        self.alpha = float(params.alpha)
        self.gamma = float(params.gamma)
        self.focal_weight = float(params.focal_weight)
        self.loc_weight = float(params.loc_weight)
        self.size_weight = float(params.size_weight)
        self.angle_weight = float(params.angle_weight)
        self.heading_weight = float(params.heading_weight)
        self.class_weight = float(params.class_weight)

    def forward(self, )
    