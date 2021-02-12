import torch
import numpy as np
import torch.nn as nn
from config import Parameters

class pillar_feature_net(nn.Module):
    """
    Point Pillar Net
    batch size: defalt = 4
    """

    def __init__(self):
        super(pillar_feature_net, self).__init__()
        self.name = 'Pillar Feature Net'
        self.conv1 = nn.Conv2d(7, 64,kernel_size=1)
        
        
    def forward(self, x): 
        x = self.conv1(x)
        return x
        


if __name__=="__main__":
    net = pillar_feature_net().double()
    test_var = np.random.randn(4, 7, 12000, 100)
    test_var = torch.from_numpy(test_var)
    test_var= test_var.double()
    res = net(test_var)
    print(res.shape)