import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from config import Parameters
from make_pillars import create_pillars

class pillar_feature_net(nn.Module):
    """
    Pillar feature net

    Input: pointclouds of batch size

    Input Shape: batch_size * pointcloud(batch_size >= 2)

    Output Shape: batch_size * 504 * 504 * 64
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


class backbone(nn.Module):
    """
    Backbone (2D CNN)

    Input Shape: batch_size * 504 * 504 * 64

    Output Shape: batch_size * 252 * 252 * 384
    """
    def __init__(self):
        super(backbone, self).__init__()

        # top-down

        self.block1 = [] # (S, 4, C)
        self.bn1 = []
        self.relu1 = []
        for i in range(4):
            if i==0:
                stride = (2,2)
                self.block1.append(nn.Conv2d(64, 64, kernel_size=(3,3), stride=stride,padding=(3,3)))
            else:
                stride = (1,1)
                self.block1.append(nn.Conv2d(64, 64, kernel_size=(3,3), stride=stride,padding=(1,1)))
            self.bn1.append(nn.BatchNorm2d(64))
            self.relu1.append(nn.ReLU())

        self.block2 = [] # (2S, 6, 2C)
        self.bn2 = []
        self.relu2 = []
        for i in range(6):
            if i==0:
                stride = (2,2) 
                self.block2.append(nn.Conv2d(64, 64*2, kernel_size=(3,3),stride=stride,padding=(3,3)))
            else:
                stride = (1,1)
                self.block2.append(nn.Conv2d(64*2, 64*2, (3,3),stride,padding=(1,1)))
            self.bn2.append(nn.BatchNorm2d(64*2))
            self.relu2.append(nn.ReLU())
        
        self.block3 = [] # (4S, 6, 4C)
        self.bn3 = []
        self.relu3 = []
        for i in range(6):
            if i==0:
                stride = (2,2) 
                self.block3.append(nn.Conv2d(64, 64*4, kernel_size=(3,3),stride=stride,padding=(3,3)))
            else:
                stride = (1,1)
                self.block3.append(nn.Conv2d(64*4, 64*4, (3,3), stride,padding=(1,1)))
            self.bn3.append(nn.BatchNorm2d(64*4))
            self.relu3.append(nn.ReLU())
        
        # upsampling

        self.up1 = nn.Conv2d(64, 2*64, (3,3), (1,1)) # (S, S, 2C)
        self.bn_up1 = nn.BatchNorm2d(2*64)
        self.relu_up1 = nn.ReLU()

        self.up2 = nn.Conv2d(2*64, 2*64, (3,3), (1,1)) # (2S, S, 2C)
        self.bn_up2 = nn.BatchNorm2d(2*64)
        self.relu_up2 = nn.ReLU()

        self.up3 = nn.Conv2d(4*64, 2*64, (3,3), (1,1)) # (4S, S, 2C)
        self.bn_up3 = nn.BatchNorm2d(2*64)
        self.relu_up3 = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x0 = x

        for i in range(4):
            x = self.block1[i](x)
            x = self.relu1[i](x)
            x = self.bn1[i](x)
        x1 = x
        x = x0
        up1 = self.up1(x1)
        up1 = self.relu_up1(up1)
        up1 = self.bn_up1(up1)
        

        for i in range(6):
            x = self.block2[i](x)
            x = self.relu2[i](x)
            x = self.bn2[i](x)
        x2 = x
        x = x0
        up2 = self.up2(x2)
        up2 = self.relu_up2(up2)
        up2 = self.bn_up2(up2)

        for i in range(6):
            x = self.block3[i](x)
            x = self.relu3[i](x)
            x = self.bn3[i](x)
        x3 = x
        x = x0
        up3 = self.up3(x3)
        up3 = self.relu_up3(up3)
        up3 = self.bn_up3(up3)

        # print(up1.shape, up2.shape, up3.shape)

        concat_feature = torch.cat([up1,up2,up3],1).permute(0,2,3,1)

        return concat_feature


class detection_head(nn.Module):
    """
    Detection Head (SSD)

    Input Shape: Batch_size * 252 * 252 * 384

    Return: occ, loc, angle, size, heading, clf
    """
    def __init__(self):
        super(detection_head, self).__init__()

        nb_anchors = len(Parameters.anchor_dims) # 4
        nb_classes = int(Parameters.nb_classes)

        self.occ = nn.Conv2d(384, nb_anchors, (1,1))
        self.sigmoid1 = nn.Sigmoid()

        self.loc = nn.Conv2d(384, 3*nb_anchors, (1,1))

        self.angle = nn.Conv2d(384, nb_anchors, (1,1))

        self.sizeconv = nn.Conv2d(384, 3*nb_anchors,(1,1))

        self.heading = nn.Conv2d(384, nb_anchors,(1,1))
        self.sigmoid2 = nn.Sigmoid()

        self.clf = nn.Conv2d(384, nb_anchors*nb_classes, (1,1))


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # occupancy
        occ = self.occ(x)
        occ = self.sigmoid1(occ).permute(0, 2, 3, 1) # bacth_size * 252 * 252 * 4

        # location
        loc = self.loc(x)
        batch_size = loc.shape[0]
        loc = loc.permute(0,2,3,1).reshape(batch_size, 252, 252, 4, 3) # bacth_size * 252 * 252 * 4 * 3

        # angle
        angle = self.angle(x).permute(0, 2, 3, 1)# bacth_size * 252 * 252 * 4 

        # size
        size = self.sizeconv(x)
        size = size.permute(0, 2, 3, 1).reshape(batch_size, 252, 252, 4, 3)# bacth_size * 252 * 252 * 4 * 3

        # heading
        heading = self.heading(x)
        heading = self.sigmoid2(heading).permute(0, 2, 3, 1)# bacth_size * 252 * 252 * 4

        # clf
        clf = self.clf(x)
        clf = clf.permute(0, 2, 3, 1).reshape(batch_size, 252, 252, 4, 4)# bacth_size * 252 * 252 * 4 * 4

        return occ, loc, angle, size, heading, clf
        


if __name__=="__main__":
    mode ='test detection head'
    if mode == 'test pillar feature net':
        net = pillar_feature_net().float()
        test_var = np.random.randn(4, 12234,4)
        test_var = torch.from_numpy(test_var)
        result = net(test_var)
        print(result.shape)
    elif mode == 'test backbone':
        net = backbone().float()
        test_var = np.random.randn(4,504,504,64)
        test_var = torch.from_numpy(test_var).float()
        # print(test_var.type())
        result = net(test_var)
        print(result.shape)
    elif mode == 'test detection head':
        net = detection_head().float()
        test_var = np.random.randn(4, 252, 252, 384)
        test_var = torch.from_numpy(test_var).float()
        result = net(test_var)
        for i in range(6):
            print(result[i].shape)
        
