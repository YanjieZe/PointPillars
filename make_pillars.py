import torch
from config import Parameters as cfg
from utils import bin2pointcloud
from point_pillars import createPillars

def create_pillars(pointcloud, # n*4
                x_min = cfg.x_min,
                x_max = cfg.x_max,
                x_step = cfg.x_step,
                y_min = cfg.y_min,
                y_max = cfg.y_max,
                y_step = cfg.y_step,
                max_point_per_pillar = cfg.max_points_per_pillar,
                max_pillars = cfg.max_pillars,
                z_min = cfg.z_min,
                z_max = cfg.z_max,
                printTime = False):
    """
    function: make point cloud into pillar points and pillar indices.
    
    return: pillar_point, indices = batch_size * 12000 * 100 *7, batch_size * 12000 *3
    """
    pillars, indices = createPillars(pointcloud, 100, 12000, float(x_step), float(y_step), float(x_min), float(x_max), float(y_min), float(y_max),-1.0,1.0,printTime)
    
    return pillars, indices

def create_pillars_target():
    pass

if __name__=="__main__":
    pc = bin2pointcloud("000000.bin")# (115384,4)
    pillars, indices = create_pillars(pc)
    print(pillars.shape)# 1 * 12000 * 100 * 7
    print(indices.shape)# 1 * 12000 * 3