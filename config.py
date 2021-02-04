import numpy as np

class GridParameters:
    """
    for pedestrian！
    """
    x_min = 0.0
    x_max = 48.0
    x_step = 0.16
    
    y_min = -20
    y_max = 20
    y_step = 0.16

    z_min = -2.5
    z_max = 0.5

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 7 # 这个不知道是干啥的
    D = 9
    nb_channels = 64 # encoder network的C=64
    
    downscaling_factor = 2 # 这个不知道干啥的

    # Length, width, height, z-center, orientation
    # car, car, pedstrian, cyclist
    anchor_dims =  np.array(
        [[3.9,1.6,1.56,-1,0],
        [3.9,1.6,1.56,-1,1.5708],
        [0.8,0.6,1.73,-0.6,0],
        [0.8,0.6,1.73,-0.6,1.5708]],dtype=np.float32
    ).tolist()

    positive_iou_threshold = 0.6
    negative_iou_threshold = 0.3
    batch_size = 4
    total_traning_epochs = 160

    # 下面的是loss function和optim的参数
    learning_rate = 2e-4
    decay_rate = 1e-8
    L1 = 0
    L2 = 0
    alpha = 0.25
    gamma = 2.0

