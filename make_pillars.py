import torch
from config import GridParameters as cfg
from utils import bin2pointcloud

def make_pillars(pointcloud, 
                x_min = cfg.x_min,
                x_max = cfg.x_max,
                x_step = cfg.x_step,
                y_min = cfg.y_min,
                y_max = cfg.y_max,
                y_step = cfg.y_step,
                max_point_per_pillar = cfg.max_points_per_pillar,
                max_pillars = cfg.max_pillars):

    points_num = pointcloud.shape[0]
    for i  


if __name__=="__main__":
    pc = bin2pointcloud("000000.bin")
    make_pillars(pc)