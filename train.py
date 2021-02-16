import torch
from processor import kitti_dataset
from network import point_pillars_net
from lossfunction import point_pillars_loss
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Parameters


def train_PointPillarNet(batch_size = Parameters.batch_size,
                        learning_rate = Parameters.learning_rate,
                        decay_learning_rate = Parameters.decay_rate,
                        epoch_num = 160,
                        print_every = 10,
                        device = torch.device('cuda')
                        ):
                        
    dataset = kitti_dataset()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    if device:
        model = point_pillars_net().float().to(device)
        loss_function = point_pillars_loss().float().to(device)
    else:
        model = point_pillars_net().float()
        loss_function = point_pillars_loss().float()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_learning_rate)

    for epoch in range(epoch_num):
        model.train()
        print('epoch %d start:'%epoch)
        if device:
            torch.cuda.empty_cache()
        for idx,([pillar_points, indices], [occ0, loc0, size0, angle0, heading0, clf0]) in enumerate(dataloader):
            # print(pillar_points.shape, indices.shape)
            print('processing %d ...'%idx)
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



if __name__=='__main__':
    
    train_PointPillarNet(batch_size=2, device=None)

   
