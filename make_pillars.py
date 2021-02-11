import torch
from config import GridParameters as cfg
from utils import bin2pointcloud
from point_pillars import createPillars

def make_pillars(pointcloud, # n*4
                x_min = cfg.x_min,
                x_max = cfg.x_max,
                x_step = cfg.x_step,
                y_min = cfg.y_min,
                y_max = cfg.y_max,
                y_step = cfg.y_step,
                max_point_per_pillar = cfg.max_points_per_pillar,
                max_pillars = cfg.max_pillars,
                z_min = -1.0,
                z_max = 1.0,
                printTime = False):

    pillars = createPillars(pointcloud, 100, 12000, float(x_step), float(y_step), float(x_min), float(x_max), float(y_min), float(y_max),-1.0,1.0,printTime)
    
    return pillars


if __name__=="__main__":
    pc = bin2pointcloud("000000.bin")# (115384,4)
    pillars = make_pillars(pc)
    print(pillars[1])# 2*1*1