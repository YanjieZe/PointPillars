import torch
from processor import kitti_dataset
from network import point_pillars_net
from lossfunction import point_pillars_loss
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Parameters
import torch.nn as nn
import torch.nn.init as init

def train_PointPillarNet(batch_size = Parameters.batch_size,
                        learning_rate = Parameters.learning_rate,
                        decay_learning_rate = Parameters.decay_rate,
                        epoch_num = 1,
                        print_every = 10,
                        device = None,
                        weight_init = None
                        ):
                        
    dataset = kitti_dataset()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    if device:
        model = point_pillars_net(device).float()
        loss_function = point_pillars_loss().float().to(device)
    else:
        model = point_pillars_net().float()
        loss_function = point_pillars_loss().float()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_learning_rate)

    if weight_init:
        model.apply(weight_init)
    for epoch in range(epoch_num):
        model.train()
        print('epoch %d start:'%epoch)
        if device:
            torch.cuda.empty_cache()
        for idx,([pillar_points, indices], [occ0, loc0, size0, angle0, heading0, clf0]) in enumerate(dataloader):
            # print(pillar_points.shape, indices.shape)
            # print('processing %d ...'%idx)
            if device:
                pillar_points = torch.autograd.Variable(pillar_points).float().to(device)
                indices = torch.autograd.Variable(indices).long().to(device)
            else:
                pillar_points = torch.autograd.Variable(pillar_points).float()
                indices = torch.autograd.Variable(indices).long()
            occ, loc, size, angle, heading, clf = model(pillar_points, indices)
            loss = loss_function(loc, size, angle, clf, heading, loc0, size0, angle0, clf0, heading0)
            print('loss: ', loss.item())
            if (idx+1)%print_every==0:
                print('idx: %d, loss: %d'%(idx, loss.item()))
            
            optimizer.zero_grad()
            loss.backward
            optimizer.step()
    # 保存模型参数
    torch.save(net.state_dict(),'pillar_net_params.pkl')



def weight_init(m):
    """设计参数初始化"""
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

if __name__=='__main__':
    # device = torch.device('cuda')
    train_PointPillarNet(batch_size=4)

   
