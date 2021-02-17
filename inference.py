from network import point_pillars_net

def inference(pointcloud, device=None, ):
    model = point_pillars_net(device)