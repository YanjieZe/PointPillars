import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from config import Parameters
from make_pillars import create_pillars

class pillar_feature_net(nn.Module):
    """
    Point Pillar Net

    Input: batch_size * 7 * 12000 * 100(batch_size >= 2)

    Output: batch_size * 64 * 12000 * 100
    """

    def __init__(self):
        super(pillar_feature_net, self).__init__()
        self.name = 'Pillar Feature Net'
        self.conv1 = nn.Conv2d(7, 64,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d((1,100))
        
        
    def forward(self, pointclouds):

        pillar_points, pillar_indices = create_pillars(pointclouds[0])  
        pillar_points = torch.from_numpy(pillar_points)
        pillar_indices = torch.from_numpy(pillar_indices).long()
        # a = pillar_indices[0,0,:].numpy()
        # print(a[0])
        if pointclouds.shape[0] >1:
            i = 1
            while i < pointclouds.shape[0]:
                pillar_points_new, pillars_indices_new = create_pillars(pointclouds[i])
                pillar_points_new = torch.from_numpy(pillar_points_new).long()
                pillars_indices_new = torch.from_numpy(pillars_indices_new)
                pillar_points = torch.cat([pillar_points, pillar_points_new], 0)
                pillar_indices = torch.cat([pillar_indices, pillars_indices_new], 0)
                i += 1
        
        x = pillar_points.permute(0, 3, 1, 2) # Batch_size * 7 * 12000 * 100
        # print(x.shape[0])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print(x.shape)
        x = self.maxpooling1(x).squeeze().permute(0,2,1) # bacth_size * 12000 * 64

        pseudo_img = self.scatter_img(pillar_features=x, pillar_indices=pillar_indices)

        return pseudo_img
    

    def scatter_img(self, pillar_features, pillar_indices):
        """
        Function: scatter pillar feature into a pseudo-img

        Input Shape: Batch_size * 12000 * 64, Batch_size * 12000 * 3

        Output Shape: Batch_size * width * height * 64
        """
        width = int((Parameters.x_max - Parameters.x_min)/Parameters.x_step)
        height = int((Parameters.y_max - Parameters.y_min)/Parameters.y_step)
        # print(width, height)
        batch_size = pillar_features.shape[0]
        pseudo_img = torch.zeros(batch_size, width, height, 64)
        
        i = 0
        # Loop to scatter feature into pseudo-img
        while i < batch_size:
            img = torch.zeros(width, height, 64)

            j = 0
            while j < 12000:
                pillar_coordinate = pillar_indices[i, j, :].numpy()
                if (pillar_coordinate==[0,0,0]).all():
                    j += 1
                    continue
                if pillar_coordinate[1]>=width:
                    raise Exception('Pillar Coordinate X Out of Bounds')
                if pillar_coordinate[2]>=height:
                    raise Exception('Pillar Coordinate Y Out of Bounds')
                    
                pseudo_img[i, pillar_coordinate[1], pillar_coordinate[2], :] = pillar_features[i, j, :]
                j += 1
            i += 1

        
        return pseudo_img


if __name__=="__main__":
    net = pillar_feature_net().float()
    test_var = np.random.randn(4, 12234,4)
    test_var = torch.from_numpy(test_var)
    result = net(test_var)
    print(result.shape)